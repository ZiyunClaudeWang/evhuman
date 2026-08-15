import torch
import scipy
from torch.utils.data import DataLoader
import os
import time
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
        num_workers=args.num_workers,
        pin_memory=False
    )

    # set model
    from pl_models.event_human_model import EventHumanModel
    model = EventHumanModel(args)
    model = model.to(device=device)
    if args.model_dir is not None:
        print('[model dir] model loaded from %s' % args.model_dir)
        checkpoint = torch.load(args.model_dir, map_location=device)
        ckpt_state = checkpoint['model_state_dict']
        model_state = model.state_dict()
        filtered = {k: v for k, v in ckpt_state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        skipped = set(ckpt_state.keys()) - set(filtered.keys())
        if skipped:
            print('[info] Skipped keys (shape mismatch or missing):', skipped)
        model_state.update(filtered)
        model.load_state_dict(model_state)
    else:
        raise ValueError('Cannot find trained model %s.' % args.model_dir)

    print('------------------------------------- test ------------------------------------')
    start_time = time.time()
    model.eval()
    results = collections.defaultdict(list)

    if args.test_high_fps:
        test_length = args.skip * 8
        model.resize_smpl(args.batch_size * (test_length + 1))

    keypoint_folder = os.path.join("tmp/keypoints", args.model_dir.replace("/", "_").replace(".pkl", ""))

    from tqdm import tqdm
    with torch.set_grad_enabled(False):
        for iter, data in enumerate(tqdm(test_generator)):
            for k in data.keys():
                if k != 'info':
                    data[k] = data[k].to(device=device, dtype=dtype)

            B, T = data['events'].size()[0], data['events'].size()[1]
            init_shape = data['init_shape'][:, 0]

            if args.test_high_fps:

                test_length = args.skip * 8
                model.resize_smpl(B * (test_length + 1))
                step = 1
                decode_ts = torch.arange(start=0, end=test_length, step=step).unsqueeze(0)
                decode_ts = decode_ts / test_length * 2 - 1
                decode_ts = torch.concat([decode_ts, torch.ones_like(decode_ts[:, :1])], dim=1)

                if args.hmr_model and args.hmr_interp:
                    out = model(data, test=True)
                    pred_joints3d = out['joints3d']
                    pred_joints3d_np = pred_joints3d.cpu().numpy()

                    decode_ts_np = decode_ts[0, :-1].cpu().numpy()
                    decode_ts_np = decode_ts_np[:-8]

                    low_fps_time = torch.arange(start=0, end=args.skip, step=step)
                    low_fps_time = (low_fps_time / args.skip * 2 - 1).cpu().numpy()

                    inter = scipy.interpolate.interp1d(low_fps_time, pred_joints3d_np, axis=1)
                    pred_joints3d = inter(decode_ts_np)
                    pred_joints3d = torch.from_numpy(pred_joints3d).to(device=device, dtype=dtype).contiguous()

                    out_verts_np = out['verts'][:, 1:, ...].cpu().numpy()
                    inter = scipy.interpolate.interp1d(low_fps_time, out_verts_np, axis=1)
                    out_verts_np = inter(decode_ts_np)
                    out_verts = torch.from_numpy(out_verts_np).to(device=device, dtype=dtype).contiguous()
                    T = pred_joints3d.shape[1]

                    target_joints3d = data['high_fps_joints3d'][:, :-8, ...].contiguous()
                    gt_theta = data['high_fps_theta'][:, :-8, ...].contiguous()
                    gt_tran = data['high_fps_tran'][:, :-8, ...].contiguous()

                else:
                    out = model(data, test=True, override_clip_length=test_length+1, decode_ts=decode_ts)

                    new_test_length = out['joints3d'].size(1)
                    pred_joints3d = out['joints3d'][:, :new_test_length, ...].contiguous()
                    target_joints3d = data['high_fps_joints3d'].contiguous()[:, :new_test_length, ...]
                    gt_theta = data['high_fps_theta'].contiguous()[:, :new_test_length, ...]
                    gt_tran = data['high_fps_tran'].contiguous()[:, :new_test_length, ...]
                    out_verts = out['verts'][:, 1:new_test_length+1, ...].contiguous()
                    T = new_test_length
            else:
                out = model(data, test=True)
                pred_joints3d = out['joints3d']
                target_joints3d = data['joints3d']
                gt_theta = data['theta']
                gt_tran = data['tran']

                out_verts = out['verts'][:, 1:, ...]

            if args.eval_downsample > 1:
                t_range = torch.arange(0, T, args.eval_downsample)
                if t_range[-1] != T - 1:
                    t_range = torch.cat([t_range, torch.tensor([T-1])], axis=0)
                    t_range = t_range.to(device=device, dtype=torch.long)

                pred_joints3d = pred_joints3d[:, t_range, ...]
                target_joints3d = target_joints3d[:, t_range, ...]
                gt_theta = gt_theta[:, t_range, ...]
                gt_tran = gt_tran[:, t_range, ...]
                out_verts = out_verts[:, t_range, ...]

                T = out_verts.shape[1]

            mpjpe = compute_mpjpe(pred_joints3d.contiguous(), target_joints3d.contiguous())  # [B, T, 24]
            pa_mpjpe = compute_pa_mpjpe(pred_joints3d.contiguous(), target_joints3d.contiguous())  # [B, T, 24]
            pel_mpjpe = compute_pelvis_mpjpe(pred_joints3d.contiguous(), target_joints3d.contiguous())  # [B, T, 24]
            pck = compute_pck(pred_joints3d.contiguous(), target_joints3d.contiguous())  # [B, T, 24]
            pck_head = compute_pck_head(pred_joints3d.contiguous(), target_joints3d.contiguous())  # [B, T, 24]
            pck_torso = compute_pck_torso(pred_joints3d.contiguous(), target_joints3d.contiguous())  # [B, T, 24]

            # [B*T, 3], [B*T, 3, 3], [B*T, 3, 1]
            _, s, R, t = batch_compute_similarity_transform_torch(
                pred_joints3d.contiguous().view(-1, 24, 3), target_joints3d.contiguous().view(-1, 24, 3), True)

            s = s.unsqueeze(-1).unsqueeze(-1)
            pa_verts = s * R.bmm(out_verts.reshape(B * T, 6890, 3).permute(0, 2, 1)) + t
            pa_verts = pa_verts.permute(0, 2, 1).view(B, T, 6890, 3)
            pa_verts = torch.cat([out['verts'][:, 0:1], pa_verts], dim=1)  # [B, T+1, 6890, 3]

            # get target vertex
            if init_shape.dim() == 3:
                beta = init_shape[:, :, 75:85].repeat(1, T, 1)  # [B, T, 10]
            else:
                beta = init_shape[:, None, 75:85].repeat(1, T, 1)  # [B, T, 10]

            target_verts, target_joints3d_rendered, _ = model.smpl(
                beta=beta.view(-1, 10),
                theta=gt_theta.contiguous().view(-1, 72),
                get_skin=True)
            target_verts = target_verts.view(B, T, target_verts.size(1), target_verts.size(2)) + gt_tran

            target_verts = torch.cat([out['verts'][:, 0:1], target_verts], dim=1)  # [B, T+1, 6890, 3]

            pve = torch.mean(torch.sqrt(torch.sum((target_verts[:, 1:] - pa_verts[:, 1:]) ** 2, dim=-1)), dim=-1)  # [B, T]

            # collect results
            results['scalar/mpjpe'].append(mpjpe.detach())
            results['scalar/pa_mpjpe'].append(pa_mpjpe.detach())
            results['scalar/pel_mpjpe'].append(pel_mpjpe.detach())
            results['scalar/pck'].append(pck.detach().float())
            results['scalar/pck_head'].append(pck_head.detach().float())
            results['scalar/pck_torso'].append(pck_torso.detach().float())
            results['scalar/pve'].append(pve.detach())

            if args.save_keypoints:
                tran = out['trans']
                if not os.path.exists(keypoint_folder):
                    os.makedirs(keypoint_folder)
                for bb in range(B):
                    action = data['info'][0][bb]
                    init_idx = data['info'][1][bb][0].item()
                    save_name = "{}_{:04d}".format(action, init_idx)
                    np.savez_compressed(
                        os.path.join(keypoint_folder, "{}_from_{}_keypoints".format(save_name, init_idx)),
                        pred_keypoints=pred_joints3d[bb].cpu().numpy(),
                        gt_keypoints=target_joints3d[bb].cpu().numpy(),
                        tran=tran[bb].cpu().numpy())

            if iter % 200 == 0:
                results['mpjpe'] = torch.mean(torch.cat(results['scalar/mpjpe'], dim=0), dim=(0, 1, 2))
                results['pa_mpjpe'] = torch.mean(torch.cat(results['scalar/pa_mpjpe'], dim=0), dim=(0, 1, 2))
                results['pel_mpjpe'] = torch.mean(torch.cat(results['scalar/pel_mpjpe'], dim=0), dim=(0, 1, 2))
                results['pck'] = torch.mean(torch.cat(results['scalar/pck'], dim=0), dim=(0, 1, 2))
                results['pck_head'] = torch.mean(torch.cat(results['scalar/pck_head'], dim=0), dim=(0, 1, 2))
                results['pck_torso'] = torch.mean(torch.cat(results['scalar/pck_torso'], dim=0), dim=(0, 1, 2))
                results['pve'] = torch.mean(torch.cat(results['scalar/pve'], dim=0))

                end_time = time.time()
                time_used = (end_time - start_time) / 60.
                print('>>> time used: {:.2f} mins \n'
                    '    mpjpe {}\n'
                    '    pa_mpjpe {}\n'
                    '    pel_mpjpe {}\n'
                    '    pck {}\n'
                    '    pck_head {}\n'
                    '    pck_torso {}\n'
                    '    pve {}\n'
                    .format(time_used, 1000 * results['mpjpe'], 1000 * results['pa_mpjpe'],
                            1000 * results['pel_mpjpe'], results['pck'], results['pck_head'],
                            results['pck_torso'], 1000 * results['pve']))

        results['mpjpe'] = torch.mean(torch.cat(results['scalar/mpjpe'], dim=0), dim=(0, 1, 2))
        results['pa_mpjpe'] = torch.mean(torch.cat(results['scalar/pa_mpjpe'], dim=0), dim=(0, 1, 2))
        results['pel_mpjpe'] = torch.mean(torch.cat(results['scalar/pel_mpjpe'], dim=0), dim=(0, 1, 2))
        results['pck'] = torch.mean(torch.cat(results['scalar/pck'], dim=0), dim=(0, 1, 2))
        results['pck_head'] = torch.mean(torch.cat(results['scalar/pck_head'], dim=0), dim=(0, 1, 2))
        results['pck_torso'] = torch.mean(torch.cat(results['scalar/pck_torso'], dim=0), dim=(0, 1, 2))
        results['pve'] = torch.mean(torch.cat(results['scalar/pve'], dim=0))

        end_time = time.time()
        time_used = (end_time - start_time) / 60.
        print('>>> time used: {:.2f} mins \n'
              '    mpjpe {}\n'
              '    pa_mpjpe {}\n'
              '    pel_mpjpe {}\n'
              '    pck {}\n'
              '    pck_head {}\n'
              '    pck_torso {}\n'
              '    pve {}\n'
              .format(time_used, 1000 * results['mpjpe'], 1000 * results['pa_mpjpe'],
                      1000 * results['pel_mpjpe'], results['pck'], results['pck_head'],
                      results['pck_torso'], 1000 * results['pve']))


def get_args():
    def print_args(args):
        """ Prints the argparse arguments applied
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
    parser.add_argument('--data_root', type=str, default='data/mmhpsd')
    parser.add_argument('--result_dir', type=str, default='outputs')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--smpl_dir', type=str, default='event_hpe/smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl')

    parser.add_argument('--events_input_channel', type=int, default=8)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--test_steps', type=int, default=8)
    parser.add_argument('--train_steps', type=int, default=8)
    parser.add_argument('--skip', type=int, default=2)

    parser.add_argument('--model_batchsize', type=int, default=16)
    parser.add_argument('--use_hmr_feats', type=int, default=1)
    parser.add_argument('--use_flow', type=int, default=1)

    parser.add_argument('--target_action', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--clip_len', type=int, default=8)
    parser.add_argument('--num_event_channels', type=int, default=8)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--direct_regress", action="store_true")
    parser.add_argument("--regress_6d", action="store_true")

    parser.add_argument("--left_mult", action="store_true")
    parser.add_argument("--pred_traj", action="store_true")
    parser.add_argument("--finetune_gmp", action="store_true")

    parser.add_argument("--no_pose0", action="store_true")
    parser.add_argument("--ours_full_pose0", action="store_true")
    parser.add_argument("--hmr_model", action="store_true")
    parser.add_argument("--hmr_interp", action="store_true")
    parser.add_argument("--backbone", type=str, default='resnet34')

    parser.add_argument('--theta_loss', type=float, default=10)
    parser.add_argument('--joints3d_loss', type=float, default=1)
    parser.add_argument('--delta_tran_loss', type=float, default=0)
    parser.add_argument('--use_geodesic_loss', type=int, default=1)
    parser.add_argument('--joints2d_loss', type=float, default=10)
    parser.add_argument('--flow_loss', type=float, default=0.1)
    parser.add_argument('--contrast_loss', type=float, default=0)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--tran_loss', type=float, default=1)
    parser.add_argument('--lr_start', '-lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_step', type=float, default=1)
    parser.add_argument("--our_data", action="store_true")
    parser.add_argument('--max_steps', type=int, default=8)

    parser.add_argument('--use_volumes', type=int, default=0)
    parser.add_argument("--event_folder", type=str, default='data/mmhpsd_events/')

    parser.add_argument("--test_high_fps", action="store_true")
    parser.add_argument("--abl_transformer", action="store_true")
    parser.add_argument("--eval_downsample", type=int, default=1)

    parser.add_argument("--save_keypoints", action="store_true")

    my_args = parser.parse_args()
    for k, v in vars(my_args).items():
        setattr(args, k, v)

    print_args(args)
    return args


def main():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    test_whole_set(args)


if __name__ == '__main__':
    main()
