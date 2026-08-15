"""Render the FPS-ladder demo: one event stream, one latent code, three
temporal resolutions.

Panels (left to right): raw events as a 3D space-time cloud, then the same
prediction decoded at 15, 120, and 240 FPS. All panels share one wall clock
in slow motion, so the 15 FPS panel visibly steps while 240 FPS is smooth —
the continuous-time motion field is queried at more timestamps, nothing is
interpolated or retrained.

Usage:
    PYTHONPATH=NeMF/src:$PYTHONPATH python render_fps_ladder.py \
        --model_dir ckpts/beahm/model_events_pose.pkl \
        --our_data --skip 8 --target_action Starjump --num_clips 3
"""
import argparse
import os

import cv2
import numpy as np
import torch

from tqdm import tqdm

from event_hpe.SMPL import batch_rodrigues
from render_demo_video import (render_mesh, render_raw_events_3d, write_video)


RATES = [15, 120, 240]          # decoded FPS per prediction panel
BASE_RATE = 240                 # video timeline resolution (frames per clip span)


def main():
    from NeMF.src.arguments import Arguments
    args = Arguments('NeMF/configs', filename='application.yaml')
    args.config_path = 'NeMF/configs'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/beahm')
    parser.add_argument('--model_dir', type=str,
                        default='ckpts/beahm/model_events_pose.pkl')
    parser.add_argument('--event_folder', type=str, default='data/beahm_events/')
    parser.add_argument('--save_folder', type=str, default='outputs/fps_ladder/')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--fps', type=int, default=30, help='playback fps')
    parser.add_argument('--num_clips', type=int, default=3)
    parser.add_argument('--resolution', type=int, default=380)
    parser.add_argument('--skip', type=int, default=8)
    parser.add_argument('--target_action', type=str, default=None)
    parser.add_argument('--our_data', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--contrast_loss', type=float, default=0)
    parser.add_argument('--test_high_fps', action='store_true')

    my_args = parser.parse_args()
    for k, v in vars(my_args).items():
        setattr(args, k, v)

    from event_hpe.final_config import apply_final_config
    apply_final_config(args)
    args.raw_events = True

    device = torch.device(f'cuda:{args.gpu_id}'
                          if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    from pl_data.event_hpe_dm import EventHPEDataModule
    dm = EventHPEDataModule(args.data_root, batch_size=1, num_workers=0,
                            target_action=args.target_action, args=args)
    dataset = dm.val_db

    from pl_models.event_human_model import EventHumanModel
    model = EventHumanModel(args)
    checkpoint = torch.load(args.model_dir, map_location=device)
    ckpt_state = checkpoint['model_state_dict']
    model_state = model.state_dict()
    model_state.update({k: v for k, v in ckpt_state.items()
                        if k in model_state and v.shape == model_state[k].shape})
    model.load_state_dict(model_state)
    model = model.to(device).eval()

    os.makedirs(args.save_folder, exist_ok=True)

    # clip span in seconds at the native 120 FPS source
    span_s = args.clip_len * args.skip / 120.0
    # decoded queries per clip for each rate, and for the video timeline
    n_per_rate = {r: max(2, int(round(r * span_s))) for r in RATES}
    n_base = max(2, int(round(BASE_RATE * span_s)))
    print(f'clip span {span_s:.3f}s -> queries per clip: '
          f'{ {r: n_per_rate[r] for r in RATES} }, timeline {n_base}')

    # consecutive clips of the target action
    from collections import defaultdict
    action_clips = defaultdict(list)
    for idx, (action, frame_idx) in enumerate(dataset.all_clips):
        action_clips[action].append((idx, frame_idx))
    for action in action_clips:
        action_clips[action].sort(key=lambda x: x[1])

    span = args.clip_len * args.skip
    faces = None

    for action, clips in sorted(action_clips.items()):
        if args.target_action and args.target_action not in action:
            continue
        selected = clips[::span][:args.num_clips]
        print(f'\n=== {action}: {len(selected)} clips ===')

        # verts_by_rate[r] : list over video-timeline frames of [V, 3]
        verts_by_rate = {r: [] for r in RATES}
        gt_verts_tl = []
        event_bins = []

        for ds_idx, _ in selected:
            data = dataset[ds_idx]
            batch = {}
            for k, v in data.items():
                if k == 'info':
                    batch[k] = ([v[0]], [v[1]])
                elif isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0).to(device, dtype=dtype)
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.from_numpy(v).unsqueeze(0).to(device, dtype=dtype)
                else:
                    batch[k] = v

            # one clip, three decoding resolutions of the same latent code
            per_rate_verts = {}
            for r in RATES:
                n = n_per_rate[r]
                model.resize_smpl(n + 1)
                ts = torch.linspace(-1, 1, n).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(batch, test=True,
                                override_clip_length=n, decode_ts=ts)
                per_rate_verts[r] = out['verts'].detach().cpu().numpy()[0][1:]
                if faces is None:
                    faces = model.smpl.faces

            # GT meshes at the labelled 15 FPS keyframes
            L = args.clip_len
            init_shape = batch['init_shape']
            if init_shape.dim() == 3:
                init_shape = init_shape[:, 0]
            gt_rm = batch_rodrigues(
                batch['theta'].reshape(-1, 3)).view(L, 24, 3, 3)
            model.resize_smpl(L)
            with torch.no_grad():
                gv, _, _ = model.smpl(beta=init_shape[:, 75:85].repeat(L, 1),
                                      rotmats=gt_rm, get_skin=True)
            gt_v = (gv + batch['tran'][0, :, 0, :][:, None, :]).cpu().numpy()

            # map each timeline frame to the latest decoded pose (hold)
            for v_idx in range(n_base):
                t = v_idx / (n_base - 1)              # [0, 1] within the clip
                for r in RATES:
                    n = n_per_rate[r]
                    src = min(int(t * n), n - 1)
                    verts_by_rate[r].append(per_rate_verts[r][src])
                gt_verts_tl.append(gt_v[min(int(t * L), L - 1)])

            # raw events into timeline bins
            raw = data.get('raw_events', None)
            bins = [torch.zeros(0, 4) for _ in range(n_base)]
            if raw is not None and raw.shape[0] > 0:
                tt = raw[:, 2]
                t0, t1 = tt.min().item(), tt.max().item()
                if t1 - t0 > 1e-9:
                    slot = ((tt - t0) / (t1 - t0) * n_base).long().clamp(0, n_base - 1)
                    for j in range(n_base):
                        bins[j] = raw[slot == j]
            event_bins.extend(bins)

        if not verts_by_rate[RATES[0]]:
            continue

        # fixed camera + ground plane covering prediction and GT
        all_v = np.concatenate([np.stack(verts_by_rate[RATES[-1]]),
                                np.stack(gt_verts_tl)], axis=0)
        lo = all_v.reshape(-1, 3).min(axis=0)
        hi = all_v.reshape(-1, 3).max(axis=0)
        center = (lo + hi) / 2.0
        ground_y = float(np.median([fv[:, 1].max() for fv in gt_verts_tl]))
        span_y = max(hi[1], ground_y) - lo[1]
        scale = max(hi[0] - lo[0], span_y) / 2.0 * 1.25
        cam_y = (min(lo[1], ground_y) + max(hi[1], ground_y)) / 2.0
        d, cam_lift, res = 5.0, 0.9, args.resolution

        n_frames = len(verts_by_rate[RATES[0]])
        slowmo = n_frames / args.fps / (len(selected) * span_s)
        print(f'Rendering {n_frames} frames ({slowmo:.1f}x slow motion)...')

        n_cols = len(RATES) + 1                        # rates + GT panel
        rendered = []
        for fid in tqdm(range(n_frames)):
            # top row: the event stream, full width
            ev_row = render_raw_events_3d(event_bins, fid,
                                          img_size=args.img_size,
                                          size=(n_cols * res, res))
            cv2.putText(ev_row, 'Events', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # bottom row: predictions at each rate, then GT
            panels = []
            for r in RATES:
                img = render_mesh(verts_by_rate[r][fid], faces,
                                  [center[0], cam_y - cam_lift, center[2] - d],
                                  [center[0], cam_y, center[2]],
                                  [0, -1, 0], scale=scale, size=(res, res),
                                  ground_y=ground_y, ground_center=center)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(img, f'Ours {r} FPS', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                panels.append(img)
            gt_img = render_mesh(gt_verts_tl[fid], faces,
                                 [center[0], cam_y - cam_lift, center[2] - d],
                                 [center[0], cam_y, center[2]],
                                 [0, -1, 0], scale=scale, size=(res, res),
                                 color='wheat',
                                 ground_y=ground_y, ground_center=center)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
            cv2.putText(gt_img, 'GT 15 FPS', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            panels.append(gt_img)

            cv2.putText(panels[0], f'{action.split("_")[0]}  '
                        f'({slowmo:.0f}x slow motion)', (10, res - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            rendered.append(np.concatenate(
                [ev_row, np.concatenate(panels, axis=1)], axis=0))

        out_path = os.path.join(args.save_folder, f'{action}_fps_ladder.mp4')
        write_video(out_path, rendered, args.fps)
        print(f'  Saved: {out_path} ({n_frames} frames, '
              f'{n_frames / args.fps:.1f}s)')


if __name__ == '__main__':
    main()
