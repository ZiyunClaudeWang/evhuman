import torch
import matplotlib.pyplot as plt
import pyvista as pv
import cv2
from torch.utils.data import DataLoader
import os
import time
import sys
sys.path.append('../')
from event_hpe.loss_funcs import compute_mpjpe, compute_pa_mpjpe, compute_pelvis_mpjpe, \
    compute_pck, compute_pck_head, compute_pck_torso, batch_compute_similarity_transform_torch

import collections
import numpy as np

def test_whole_set(args):
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32
    from pl_data.event_hpe_dm import EventHPEDataModule
    dm = EventHPEDataModule(args.data_root,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            target_action=args.target_action,
                            args=args
                            )
    dataset_test = dm.val_db

    test_generator = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        pin_memory=False
    )

    # set model
    from pl_models.event_human_model import EventHumanModel
    model = EventHumanModel(args)
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    # set tensorboard
    if args.model_dir is not None:
        print('[model dir] model loaded from %s' % args.model_dir)
        checkpoint = torch.load(args.model_dir, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError('Cannot find trained model %s.' % args.model_dir)

    # check path
    save_dir = args.result_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('------------------------------------- test ------------------------------------')
    start_time = time.time()
    model.eval()  # dropout layers will not work in eval mode
    results = collections.defaultdict(list)

    # inference_length = 16
    inference_length = args.inference_length
    which_frame = 412

    with torch.no_grad():
        from tqdm import tqdm
        with torch.set_grad_enabled(False):
            assert(args.batch_size == 1)
            for iter, data in enumerate(tqdm(test_generator)):

                if iter < 200:
                    continue

                for k in data.keys():
                    if k != 'info':
                        data[k] = data[k].to(device=device, dtype=dtype)

                # benchmark_time(model, data, args)
                
                if which_frame != data['info'][1][0][0]:
                    print(data['info'][1][0])
                    continue

                B, T = data['events'].size()[0], data['events'].size()[1]

                # collect result
                length = inference_length
                step = 1
                decode_ts = torch.arange(start=0, end=length, step=step).unsqueeze(0)
                decode_ts = decode_ts / length * 2 - 1
                decode_ts = torch.concat([decode_ts, torch.ones_like(decode_ts[:, :1])], dim=1)

                out = model(data, test=True, override_clip_length=inference_length+1, decode_ts=decode_ts)

            
                # save result vertices and faces as ply files
                folder = os.path.join(args.save_folder, "mesh_results")
                if not os.path.exists(folder):
                    os.makedirs(folder)

                verts = out['verts'].cpu().numpy()[:, 1:, :, :]
                faces = model.smpl.faces

                total_shift = 10
                # batch_breaks = np.cumsum(data['event_size'].cpu().numpy()).astype(int)

                for i in range(B):
                    # pred_images = model.plot_human(data, out)

                    # b_start = 0 if i == 0 else batch_breaks[i-1]
                    # events = data['raw_events'][0, b_start:b_start + int(data['event_size'][i].item())]
                    # plot_events(events.cpu().numpy())

                    to_plot = []
                    for j in tqdm(range(inference_length + 1)):
                        frac = j / inference_length

                        shifted_verts = verts[i, j] + np.array([[0, 0, frac * total_shift]])
                        # save_ply(shifted_verts, faces, filename=os.path.join(folder, 'result_%d_%d.ply' % (which_frame, j)))
                        to_plot.append({"vertices": shifted_verts, "faces": faces})

                    plot_meshes(to_plot, save_path=os.path.join(folder, 'result_{:05d}_fps_{:05d}.png'.format(which_frame, inference_length)))
                        # cv2.imwrite(os.path.join(folder, 'result_%d_%d.png' % (which_frame, j)), pred_images[j].permute(1, 2, 0).numpy()[:, :, ::-1])
                return

def benchmark_time(model, data, args):
    device = model.device
    NUM_SAMPLES = 20
    time_start = time.time()
    lengths = [4, 8, 16, 32, 64, 128, 256, 512]
    for ll in lengths:
        if args.our_data:
            smpl_dir = 'event_hpe/smpl_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
        else:
            smpl_dir = 'event_hpe/smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        from event_hpe.SMPL import SMPL
        new_smpl = SMPL(smpl_dir, ll).to(device=device)
        model.smpl = new_smpl
        for _ in range(NUM_SAMPLES):
            if args.use_flow:
                # out = model(data['events'], init_shape, data['flows'], data['hidden_feats'][:, 0])
                out = model(data, test=True, override_clip_length=ll)
            else:
                out = model(data, test=True)
        time_end = time.time()
        per_iter_time = (time_end - time_start) / NUM_SAMPLES
        print('time used: {:04f} for length {}'.format(per_iter_time, ll))
    exit()

def plot_events(events):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    events = events[::100, :]
    t = (events[:, 2] - events[0, 2]) / (events[-1, 2] - events[0, 2])

    x = events[:, 0] / 256 * 480
    y = events[:, 1] / 256 * 640

    color = (np.random.rand(events.shape[0]) > 0.5).astype(float)

    ax.scatter(t, x, y, cmap='bwr', c=color, s=5)

    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.tick_params(axis='z', which='major', labelsize=12)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.set_xlabel("t", fontsize=24, labelpad=10)
    ax.set_ylabel("x", fontsize=24, labelpad=10)
    ax.set_zlabel("y", fontsize=24, labelpad=10)
    plt.show()


def plot_meshes(meshes, save_path=None, show=False, 
                                camera_position=[-12, 0, 8.5], 
                                focal_point=[0, 0, 8.5], 
                                view_up=[0, -1, 0],
                                view_angle=30):

    if show:
        plotter = pv.Plotter(window_size=[800, 600])
    else:
        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])

    for mid, mesh in enumerate(meshes):
        vertices = mesh["vertices"]
        # faces = np.hstack([[3] + face for face in mesh["faces"]]).ravel()
        faces = np.concatenate((np.full((mesh['faces'].shape[0], 1), 3), mesh["faces"]), axis=1)
        faces = np.hstack(faces).ravel()
        pv_mesh = pv.PolyData(vertices, faces)
        if mid > 0:
            plotter.add_mesh(pv_mesh, color="lightgray", show_edges=False, opacity=0.3)
        else:
            plotter.add_mesh(pv_mesh, color="lightgray", show_edges=False)


        # Set the camera position
    # camera_position = [-12, 0, 8.5]    # Camera position in space
    # focal_point = [0, 0, 8.5]        # Point the camera is looking at
    # view_up = [0, -1, 0]            # Up direction of the camera

    plotter.camera.parallel_projection = True
    # plotter.camera.fov = view_angle
    # plotter.camera.zoom(8)
    plotter.camera.parallel_scale = 2

    # Apply the camera settings
    plotter.camera_position = [camera_position, focal_point, view_up]

    # plotter.camera.fov = 30
    plotter.camera.clipping_range = [1, 30]

    if show:
        plotter.show()

    # if save_path is not None:
    img = plotter.screenshot(save_path, scale=2, transparent_background=True, return_img=True)
    plotter.close()

    return img

def get_args():
    def print_args(args):
        """ Prints the argparse argmuments applied
        Args:
         ll - args = parser.parse_args()
        """
        _args = vars(args)
        max_length = max([len(k) for k, _ in _args.items()])
        for k, v in _args.items():
            print(' ' * (max_length - len(k)) + k + ': ' + str(v))

    # import argparse
    # parser = argparse.ArgumentParser()
    import argparse
    from NeMF.src.arguments import Arguments

    args = Arguments('NeMF/configs', filename="application.yaml")
    args.config_path = 'NeMF/configs'
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--data_root', type=str, default='data/mmhpsd')
    parser.add_argument('--result_dir', type=str, default='outputs')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--save_folder', type=str, default='tmp/')

    parser.add_argument('--test_steps', type=int, default=8)
    parser.add_argument('--train_steps', type=int, default=8)
    parser.add_argument('--skip', type=int, default=2)

    parser.add_argument('--model_batchsize', type=int, default=16)  # batch size used when training the model

    parser.add_argument('--target_action', type=str, default='subject01_group1_time1')
    # parser.add_argument('--target_action', type=str, default=None)
    parser.add_argument('--save_results', type=int, default=0)

    parser.add_argument('--smpl_dir', type=str, default='../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_worker', type=int, default=4)

    # custom
    parser.add_argument("--event_hpe_model", action="store_true")

    parser.add_argument("--test_trans_regress", action="store_true")
    parser.add_argument("--pred_seq_trans", action="store_true")
    parser.add_argument("--use_transformer", action="store_true")
    parser.add_argument("--use_h5", action="store_true")
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--residual_trans", action="store_true")
    parser.add_argument("--debug_rotmat", action="store_true")
    parser.add_argument("--finetune_gmp", action="store_true")

    # args = parser.parse_args()
    # print_args(args)
    # return args

    # IGNORE
    parser.add_argument('--theta_loss', type=float, default=10)
    parser.add_argument('--joints3d_loss', type=float, default=1)
    parser.add_argument('--joints2d_loss', type=float, default=10)
    parser.add_argument('--flow_loss', type=float, default=0.1)
    parser.add_argument('--contrast_loss', type=float, default=0)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--tran_loss', type=float, default=1)
    parser.add_argument('--lr_start', '-lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_step', type=float, default=1)
    parser.add_argument("--our_data", action="store_true")
    parser.add_argument("--event_folder", type=str, default='data/mmhpsd_events/')

    parser.add_argument('--inference_length', type=int, default=8)

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
    test_whole_set(args)
    # test_simple_instance(args)

    # data = TrackingTestDataloader(
    #     train_steps=8,
    #     test_steps=8,
    #     skip=2,
    #     events_input_channel=8,
    #     img_size=256,
    #     mode='test',
    #     use_flow=True,
    #     use_hmr_feats=False,
    #     target_action=None,
    #     use_vibe_init=True
    # )
    # # print(data.all_clips)
    # sample = data[1000]
    # for k, v in sample.items():
    #     if k is not 'info':
    #         print(k, v.size())


if __name__ == '__main__':
    main()
