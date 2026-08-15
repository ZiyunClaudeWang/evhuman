import torch
from torchvision.utils import flow_to_image
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import cv2
from tensorboardX import SummaryWriter
import sys
from event_hpe.loss_funcs import compute_losses, compute_mpjpe, compute_pa_mpjpe, compute_pelvis_mpjpe, compute_pck
import collections
import numpy as np

from pl_data.event_hpe_dm import EventHPEDataModule
from pl_models.event_human_model import EventHumanModel
from event_hpe.human_losses import batch_rodrigues
from NeMF.src.rotations import matrix_to_axis_angle, matrix_to_rotation_6d
from event_hpe.geometry import rot6d_to_rotmat
from NeMF.src.utils import estimate_angular_velocity, estimate_linear_velocity

from torch.utils.data.dataloader import default_collate

def custom_collate_fn(batch):
    # Separate out the batch by keys
    batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

    # Apply custom behavior for a specific key, e.g., 'special_key'
    if 'raw_events' in batch_dict:
        # Custom behavior for 'special_key', e.g., concatenating along dim=0
        batch_dict['raw_events'] = torch.cat(batch_dict['raw_events'], dim=0)
    
    # Use default_collate for other keys
    for key in batch_dict:
        if key != 'raw_events':
            batch_dict[key] = default_collate(batch_dict[key])

    return batch_dict


def train(args):

    torch.manual_seed(0)
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32


    dm = EventHPEDataModule(args.data_root,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            target_action=args.target_action,
                            args=args
                            )
    model = EventHumanModel(args)

    dataset_train = dm.train_db
    dataset_val = dm.val_db

    val_generator = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=custom_collate_fn
    )

    train_generator = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=custom_collate_fn
    )
    total_iters = len(dataset_train) // args.batch_size + 1

    mse_func = torch.nn.MSELoss()
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr_start)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_decay_step], gamma=args.lr_decay_rate)
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    gmp_model = model.gmp.to(device=device)

    # set tensorboard
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    writer = SummaryWriter('%s/%s/%s' % (args.result_dir, args.log_dir, start_time))
    print('[tensorboard] %s/%s/%s' % (args.result_dir, args.log_dir, start_time))
    if args.model_dir is not None:
        print('[model dir] model loaded from %s' % args.model_dir)
        checkpoint = torch.load(args.model_dir, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in checkpoint.keys() and not args.pred_traj:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    save_dir = '%s/%s/%s' % (args.result_dir, args.log_dir, start_time)
    model_dir = '%s/%s/%s/model_events_pose.pkl' % (args.result_dir, args.log_dir, start_time)

    # training
    best_loss = 1e4
    PLOT_STEP = 100
    for epoch in range(args.epochs):
        print('====================================== Epoch %i ========================================' % (epoch + 1))
        # '''
        print('------------------------------------- Training ------------------------------------')
        model.train()
        results = collections.defaultdict(list)
        # results['faces'] = model.smpl.faces
        # results['cam_intr'] = model.cam_intr
        start_time = time.time()

        for iter, data in enumerate(tqdm(train_generator)):
            # data: {events, flows, init_shape, hidden_feats, theta, tran, joints2d, joints3d, info}
            for k in data.keys():
                if k != 'info':
                    data[k] = data[k].to(device=device, dtype=dtype)

            loss_dict = dict()
            optimizer.zero_grad()
            loss = 0.
            with torch.cuda.amp.autocast():
                # compute contrast loss
                B = data['events'].shape[0]
                decode_length = args.clip_len
                L = args.clip_len
                theta_rotmat = batch_rodrigues(data['theta'].view(-1, 3)).view(data['theta'].shape[0], data['theta'].shape[1], 24, 3, 3)
                gmp_input_rotmat = theta_rotmat.view(B, decode_length, 24, 3, 3)
                step = 1
                dt = 1.0 / (args.data.fps * decode_length / L) * step

                n_joints = 24
                pos_recon, _ = model.fk(gmp_input_rotmat.view(-1, n_joints, 3, 3))  # (B x T, J, 3)
                pos_recon = pos_recon.contiguous().view(B, -1, n_joints, 3)  # (B, T, J, 3)

                # rotate the translation, still in nemf coordinate
                gmp_data = dict()
                gmp_data['rot6d'] = matrix_to_rotation_6d(gmp_input_rotmat.view(B, decode_length, 24, 3, 3))  # (B, T, J, 6)
                gmp_data['angular'] = estimate_angular_velocity(gmp_input_rotmat.clone().view(B, decode_length, 24, 3, 3), dt)
                gmp_data['pos'] = pos_recon
                gmp_data['velocity'] = estimate_linear_velocity(pos_recon, dt=dt)

                # predict relative translation in the nemf coordinate (world)
                gt_trans = data['tran']
                origin = gt_trans[:, 0, 0, :]

                gmp_data['origin'] = origin
                pred_data = gmp_model.predict(gmp_data, dt=dt, no_height=True)
                trans = pred_data['trans'][:, :, None, ...]

                tran_loss = mse_func(trans, data['tran'])
                tran_loss = args.tran_loss * tran_loss

                loss_dict['tran'] = tran_loss
                loss += tran_loss


            # scale the loss and calls backward() to create scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            # collect results
            results['scalar/tran'].append(loss_dict['tran'].detach())

            # if iter > 10:
            #     break
            # if iter % 2 == 0:
            display = 200
            if iter % (total_iters // display) == 0:
                results['tran'] = torch.mean(torch.stack(results['scalar/tran'], dim=0))
                for key in results.keys():
                    if type(results[key]) == torch.Tensor and results[key].dim() == 0: 
                        if 'debug' in key:
                            writer.add_scalar('debug/%s' % key, results[key].detach(), epoch * total_iters + iter)
                        elif 'mpjpe' in key:
                            writer.add_scalar('train/%s' % key, results[key].detach()*1000, epoch * total_iters + iter)
                        else:
                            writer.add_scalar('train/%s' % key, results[key].detach(), epoch * total_iters + iter)
                
                end_time = time.time()
                time_used = (end_time - start_time) / 60.
                print('>>> [epoch {:2d}/ iter {:6d}] \n'
                      '    , tran {:.4f}\n'
                      .format(epoch, iter,  results['tran']))
            
        print('------------------------------------- test ------------------------------------')
        start_time = time.time()
        # model.eval()  # dropout layers will not work in eval mode
        gmp_model.eval()
        results = collections.defaultdict(list)
        with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
            for iter, data in enumerate(tqdm(val_generator)):
                for k in data.keys():
                    if k != 'info':
                        data[k] = data[k].to(device=device, dtype=dtype)
                # compute contrast loss
                B = data['events'].shape[0]
                decode_length = args.clip_len
                L = args.clip_len
                theta_rotmat = batch_rodrigues(data['theta'].view(-1, 3)).view(data['theta'].shape[0], data['theta'].shape[1], 24, 3, 3)
                gmp_input_rotmat = theta_rotmat.view(B, decode_length, 24, 3, 3)
                step = 1
                dt = 1.0 / (args.data.fps * decode_length / L) * step

                n_joints = 24
                pos_recon, _ = model.fk(gmp_input_rotmat.view(-1, n_joints, 3, 3))  # (B x T, J, 3)
                pos_recon = pos_recon.contiguous().view(B, -1, n_joints, 3)  # (B, T, J, 3)

                # rotate the translation, still in nemf coordinate
                gmp_data = dict()
                gmp_data['rot6d'] = matrix_to_rotation_6d(gmp_input_rotmat.view(B, decode_length, 24, 3, 3))  # (B, T, J, 6)
                gmp_data['angular'] = estimate_angular_velocity(gmp_input_rotmat.clone().view(B, decode_length, 24, 3, 3), dt)
                gmp_data['pos'] = pos_recon
                gmp_data['velocity'] = estimate_linear_velocity(pos_recon, dt=dt)

                # predict relative translation in the nemf coordinate (world)
                gt_trans = data['tran']
                origin = gt_trans[:, 0, 0, :]

                gmp_data['origin'] = origin
                pred_data = gmp_model.predict(gmp_data, dt=dt, no_height=True)
                trans = pred_data['trans'][:, :, None, ...]

                tran_loss = mse_func(trans, data['tran'])
                tran_loss = args.tran_loss * tran_loss

                loss_dict = {}
                loss_dict['tran'] = tran_loss

                loss = 0.
                loss += tran_loss

                results['scalar/tran'].append(loss_dict['tran'].detach())
                results['scalar/loss'].append(loss.detach())

            # torch.save(model.state_dict(), model_dir)

            results['tran'] = torch.mean(torch.stack(results['scalar/tran'], dim=0))
            results['loss'] = torch.mean(torch.stack(results['scalar/loss'], dim=0))

            print('>>> [epoch {:2d}/ iter {:6d}] \n'
                    '    , tran {:.4f}\n'
                    .format(epoch, iter,  results['tran']))

            if best_loss > results['loss']:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, model_dir)
                # torch.save(model.state_dict(), model_dir)
                best_loss = results['loss']
                print('>>> Model saved as {}... best loss {:.4f}'.format(model_dir, best_loss))
            # break
        # '''
        scheduler.step()
    writer.close()


# def write_tensorboard(writer, results, epoch, progress, mode, args):
#     action, sample_frames_idx = results['info']
#     verts = results['verts'].cpu().numpy()  # [T+1, 6890, 3]
#     # cam_intr = results['cam_intr'].cpu().numpy()
#     # faces = results['faces']

#     fullpics, render_imgs = [], []
#     for i, frame_idx in enumerate(sample_frames_idx):
#         img = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.jpg' % (args.data_root, args.img_size, action, frame_idx))
#         fullpics.append(img[:, :, 0:1])

#         vert = verts[i]
#         dist = np.abs(np.mean(vert, axis=0)[2])
#         # render_img = (util.render_model(vert, faces, args.img_size, args.img_size, cam_intr, np.zeros([3]),
#         #                                 np.zeros([3]), near=0.1, far=20 + dist, img=img) * 255).astype(np.uint8)
#         # render_img = util.render_model(vert, faces, args.img_size, args.img_size, cam_intr, np.zeros([3]),
#         #                                np.zeros([3]), near=0.1, far=20 + dist, img=img)
#         # render_imgs.append(render_img)

#     fullpics = np.transpose(np.stack(fullpics, axis=0), [0, 3, 1, 2]) / 255.
#     fullpics = np.concatenate([fullpics, fullpics, fullpics], axis=1)
#     writer.add_images('%s/fullpic%06i' % (mode, epoch * 100 + progress), fullpics, 1, dataformats='NCHW')

#     # render_imgs = np.transpose(np.stack(render_imgs, axis=0), [0, 3, 1, 2])
#     # writer.add_images('%s/shape%06i' % (mode, epoch * 100 + progress), render_imgs, 1, dataformats='NCHW')


def get_args():
    def print_args(args):
        """ Prints the argparse argmuments applied
        Args:
          args = parser.parse_args()
        """
        _args = vars(args)
        max_length = max([len(k) for k, _ in _args.items()])
        for k, v in _args.items():
            print(' ' * (max_length - len(k)) + k + ': ' + str(v))

    import argparse
    from NeMF.src.arguments import Arguments

    args = Arguments('NeMF/configs', filename="application.yaml")
    args.config_path = 'NeMF/configs'

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--result_dir', type=str, default='outputs')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--smpl_dir', type=str, default='../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
    parser.add_argument('--pin_memory', type=int, default=1)
    parser.add_argument('--use_amp', type=int, default=1)

    parser.add_argument('--target_action', type=str, default=None)

    parser.add_argument('--skip', type=int, default=2)

    parser.add_argument('--delta_tran_loss', type=float, default=0)
    parser.add_argument('--tran_loss', type=float, default=1)
    parser.add_argument('--theta_loss', type=float, default=10)
    parser.add_argument('--joints3d_loss', type=float, default=1)
    parser.add_argument('--joints2d_loss', type=float, default=10)
    parser.add_argument('--flow_loss', type=float, default=0.1)
    parser.add_argument('--contrast_loss', type=float, default=0)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_start', '-lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_step', type=float, default=1)

    parser.add_argument("--data_root", type=str, default="data/mmhpsd")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--our_data", action="store_true")

    parser.add_argument("--test_trans_regress", action="store_true")
    parser.add_argument("--pred_seq_trans", action="store_true")
    parser.add_argument("--use_transformer", action="store_true")
    parser.add_argument("--use_h5", action="store_true")
    parser.add_argument("--residual_trans", action="store_true")
    parser.add_argument("--event_hpe_model", action="store_true")
    parser.add_argument("--debug_rotmat", action="store_true")
    parser.add_argument("--finetune_gmp", action="store_true")
    parser.add_argument("--rand_cut_grad", action="store_true")

    parser.add_argument("--test_high_fps", action="store_true")
    parser.add_argument("--only_train_gmp", action="store_true")
    parser.add_argument("--reset_optimizer", action="store_true")

    parser.add_argument("--event_folder", type=str, default='data/mmhpsd_events/')

    my_args = parser.parse_args()
    for k, v in vars(my_args).items():
        setattr(args, k, v)

    from event_hpe.final_config import apply_final_config
    apply_final_config(args)

    print_args(args)
    return args


def main():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    train(args)


if __name__ == '__main__':
    main()
