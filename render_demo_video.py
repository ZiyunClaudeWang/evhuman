"""Render a demo video stitching multiple consecutive clips into smooth motion.

For each test action, takes consecutive overlapping clips, decodes at high FPS,
and blends the overlap regions. Renders events + front + side views.

Usage:
    PYTHONPATH=NeMF/src:$PYTHONPATH python render_demo_video.py \
        --model_dir outputs/beahm_skip8_gmp/.../model_events_pose.pkl \
        --our_data --skip 8 --fps 30 --num_clips 10
"""
import argparse
import os

import cv2
import numpy as np
import pyvista as pv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def render_mesh(vertices, faces, position, focal_point, up, scale=1.2, size=(400, 400)):
    plotter = pv.Plotter(off_screen=True, window_size=size)
    pv_faces = np.concatenate(
        (np.full((faces.shape[0], 1), 3), faces), axis=1
    ).ravel()
    mesh = pv.PolyData(vertices, pv_faces)
    plotter.add_mesh(mesh, color='lightblue', smooth_shading=True)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = scale
    plotter.camera_position = [position, focal_point, up]
    plotter.camera.clipping_range = [0.1, 50]
    plotter.set_background('white')
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def event_to_rgb(event_vol):
    if isinstance(event_vol, torch.Tensor):
        ev = event_vol.detach().cpu().numpy()
    else:
        ev = event_vol
    if ev.ndim == 3 and ev.shape[0] <= 8:
        ev = ev.transpose(1, 2, 0)
    h, w = ev.shape[:2]
    pos = np.sum(np.clip(ev, 0, None), axis=-1)
    neg = np.sum(np.clip(-ev, 0, None), axis=-1)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if pos.max() > 0:
        img[:, :, 2] = (pos / pos.max() * 255).astype(np.uint8)
    if neg.max() > 0:
        img[:, :, 0] = (neg / neg.max() * 255).astype(np.uint8)
    return img


def event_vol_to_3d(event_vols, time_idx, total_times, size=(400, 400)):
    """Render a jAER-style 3D (x, y, t) event stream visualization.

    Takes all event volumes up to time_idx and renders the accumulated
    events as a 3D point cloud with time as the vertical axis.

    Args:
        event_vols: list of event volumes [C, H, W] (one per timestep)
        time_idx: current frame index (0 to total_times-1)
        total_times: total number of timesteps
        size: output image size
    Returns:
        img: [H, W, 3] uint8 BGR image
    """
    # Show a sliding window of events: current + 2 past frames
    window = 3
    start = max(0, time_idx - window + 1)
    end = time_idx + 1

    points = []
    colors = []

    for t_idx in range(start, end):
        if t_idx >= len(event_vols):
            break
        ev = event_vols[t_idx]
        if isinstance(ev, torch.Tensor):
            ev = ev.detach().cpu().numpy()
        if ev.ndim == 3 and ev.shape[0] <= 8:
            ev = ev.transpose(1, 2, 0)

        h, w = ev.shape[:2]
        pos = np.sum(np.clip(ev, 0, None), axis=-1)
        neg = np.sum(np.clip(-ev, 0, None), axis=-1)

        # Sample positive events
        py, px = np.where(pos > pos.max() * 0.1)
        if len(py) > 0:
            # Subsample for performance
            if len(py) > 2000:
                idx = np.random.choice(len(py), 2000, replace=False)
                py, px = py[idx], px[idx]
            t_val = (t_idx - start) / max(1, window - 1)
            pts = np.stack([px / w, t_val * np.ones(len(px)), py / h], axis=1)
            points.append(pts)
            colors.extend([(0.2, 0.3, 1.0)] * len(px))  # blue for positive

        # Sample negative events
        ny, nx = np.where(neg > neg.max() * 0.1)
        if len(ny) > 0:
            if len(ny) > 2000:
                idx = np.random.choice(len(ny), 2000, replace=False)
                ny, nx = ny[idx], nx[idx]
            t_val = (t_idx - start) / max(1, window - 1)
            pts = np.stack([nx / w, t_val * np.ones(len(nx)), ny / h], axis=1)
            points.append(pts)
            colors.extend([(1.0, 0.2, 0.2)] * len(nx))  # red for negative

    if not points:
        return np.ones((*size, 3), dtype=np.uint8) * 255

    all_points = np.concatenate(points, axis=0)
    all_colors = np.array(colors)

    plotter = pv.Plotter(off_screen=True, window_size=size)
    cloud = pv.PolyData(all_points)
    cloud['colors'] = (all_colors * 255).astype(np.uint8)
    plotter.add_points(cloud, scalars='colors', rgb=True, point_size=2)

    # Camera: slight angle from the side to see the time axis
    plotter.camera_position = [
        (0.5, -0.8, 0.5),   # camera position
        (0.5, 0.5, 0.5),    # focal point (center of cube)
        (0, 0, -1),          # up vector (y points into screen, z is vertical)
    ]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 0.7
    plotter.set_background('white')

    # Add time axis label
    plotter.add_text('t', position=(20, size[1] - 40), font_size=12, color='black')

    img = plotter.screenshot(return_img=True)
    plotter.close()

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    from NeMF.src.arguments import Arguments
    args = Arguments('NeMF/configs', filename='application.yaml')
    args.config_path = 'NeMF/configs'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='data/beahm')
    parser.add_argument('--model_dir', type=str,
                        default='ckpts/beahm/model_events_pose.pkl')
    parser.add_argument('--event_folder', type=str,
                        default='data/beahm_events/')
    parser.add_argument('--save_folder', type=str, default='outputs/demo_videos/')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--num_clips', type=int, default=8)
    parser.add_argument('--resolution', type=int, default=400)
    parser.add_argument('--skip', type=int, default=8)
    parser.add_argument('--hmr_model', action='store_true', default=True)
    parser.add_argument('--ours_full_pose0', action='store_true', default=True)
    parser.add_argument('--left_mult', action='store_true', default=True)
    parser.add_argument('--pred_traj', action='store_true', default=True)
    parser.add_argument('--our_data', action='store_true', default=True)
    parser.add_argument('--abl_transformer', action='store_true')
    parser.add_argument('--no_pose0', action='store_true')
    parser.add_argument('--use_hmr_feats', type=int, default=0)
    parser.add_argument('--use_flow', type=int, default=1)
    parser.add_argument('--use_volumes', type=int, default=0)
    parser.add_argument('--use_geodesic_loss', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet34')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--clip_len', type=int, default=8)
    parser.add_argument('--num_event_channels', type=int, default=8)
    parser.add_argument('--events_input_channel', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=8)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--contrast_loss', type=float, default=0)
    parser.add_argument('--test_high_fps', action='store_true')
    parser.add_argument('--target_action', type=str, default=None)

    my_args = parser.parse_args()
    for k, v in vars(my_args).items():
        setattr(args, k, v)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    from pl_data.event_hpe_dm import EventHPEDataModule
    dm = EventHPEDataModule(args.data_root, batch_size=1, num_workers=0,
                            target_action=args.target_action, args=args)
    dataset = dm.val_db

    from pl_models.event_human_model import EventHumanModel
    args.batch_size = 1
    model = EventHumanModel(args)
    model = model.to(device)

    print(f'Loading model from {args.model_dir}')
    checkpoint = torch.load(args.model_dir, map_location=device)
    ckpt_state = checkpoint['model_state_dict']
    model_state = model.state_dict()
    filtered = {k: v for k, v in ckpt_state.items()
                if k in model_state and v.shape == model_state[k].shape}
    model_state.update(filtered)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    os.makedirs(args.save_folder, exist_ok=True)

    # Group clips by action
    from collections import defaultdict
    action_clips = defaultdict(list)
    for idx, (action, frame_idx) in enumerate(dataset.all_clips):
        action_clips[action].append((idx, frame_idx))

    # Sort each action's clips by frame index for temporal continuity
    for action in action_clips:
        action_clips[action].sort(key=lambda x: x[1])

    faces = None
    all_action_frames = {}

    for action, clips in sorted(action_clips.items()):
        if args.target_action and args.target_action not in action:
            continue

        # Take evenly spaced clips for coverage
        total = len(clips)
        step = max(1, total // args.num_clips)
        selected = clips[::step][:args.num_clips]

        print(f'\n=== {action}: {len(selected)} clips ===')

        action_verts = []
        action_events = []  # store raw event volumes for 3D rendering

        for clip_idx, (ds_idx, frame_idx) in enumerate(selected):
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

            events = data['events']  # [T, C, H, W]
            L = events.shape[0]

            model.resize_smpl(L + 1)

            with torch.no_grad():
                out = model(batch, test=True)

            verts = out['verts'].detach().cpu().numpy()[0]  # [T+1, V, 3]
            if faces is None:
                faces = model.smpl.faces

            # Skip the init frame (t=0), keep predicted frames
            for t in range(1, verts.shape[0]):
                action_verts.append(verts[t])
                ev_idx = min(t - 1, L - 1)
                action_events.append(events[ev_idx])  # store raw volume

        if not action_verts:
            continue

        all_action_frames[action] = (action_verts, action_events)
        print(f'  {len(action_verts)} total frames')

    # Render all actions
    res = args.resolution
    for action, (verts_list, event_list) in all_action_frames.items():
        print(f'\nRendering {action}: {len(verts_list)} frames...')
        rendered = []
        for fid in tqdm(range(len(verts_list))):
            v = verts_list[fid]
            c = v.mean(axis=0)
            d = 5.0

            front = render_mesh(v, faces,
                                [c[0], c[1], c[2] - d], [c[0], c[1], c[2]],
                                [0, -1, 0], scale=1.2, size=(res, res))
            side = render_mesh(v, faces,
                               [c[0] + d, c[1], c[2]], [c[0], c[1], c[2]],
                               [0, -1, 0], scale=1.2, size=(res, res))

            ev_img = event_vol_to_3d(event_list, fid, len(verts_list), size=(res, res))
            front_bgr = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
            cv2.putText(front_bgr, 'Front', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            side_bgr = cv2.cvtColor(side, cv2.COLOR_RGB2BGR)
            cv2.putText(side_bgr, 'Side', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Add action label
            cv2.putText(front_bgr, action.split('_')[0], (10, res - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            frame = np.concatenate([ev_img, front_bgr, side_bgr], axis=1)
            rendered.append(frame)

        video_path = os.path.join(args.save_folder, f'{action}.mp4')
        h, w = rendered[0].shape[:2]
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
        for f in rendered:
            writer.write(f)
        writer.release()
        print(f'  Saved: {video_path} ({len(rendered)} frames, {len(rendered)/args.fps:.1f}s)')

    # Combine all actions into one compilation video
    print('\nCreating compilation video...')
    all_frames = []
    for action in sorted(all_action_frames.keys()):
        verts_list, event_list = all_action_frames[action]
        for fid in tqdm(range(len(verts_list)), desc=action):
            v = verts_list[fid]
            c = v.mean(axis=0)
            d = 5.0

            front = render_mesh(v, faces,
                                [c[0], c[1], c[2] - d], [c[0], c[1], c[2]],
                                [0, -1, 0], scale=1.2, size=(res, res))
            side = render_mesh(v, faces,
                               [c[0] + d, c[1], c[2]], [c[0], c[1], c[2]],
                               [0, -1, 0], scale=1.2, size=(res, res))

            ev_img = event_vol_to_3d(event_list, fid, len(verts_list), size=(res, res))
            front_bgr = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
            cv2.putText(front_bgr, 'Front', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            side_bgr = cv2.cvtColor(side, cv2.COLOR_RGB2BGR)
            cv2.putText(side_bgr, 'Side', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(front_bgr, action.split('_')[0], (10, res - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            frame = np.concatenate([ev_img, front_bgr, side_bgr], axis=1)
            all_frames.append(frame)

    comp_path = os.path.join(args.save_folder, 'compilation.mp4')
    h, w = all_frames[0].shape[:2]
    writer = cv2.VideoWriter(comp_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    for f in all_frames:
        writer.write(f)
    writer.release()
    print(f'Saved compilation: {comp_path} ({len(all_frames)} frames, {len(all_frames)/args.fps:.1f}s)')


if __name__ == '__main__':
    main()
