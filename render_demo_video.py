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

from event_hpe.SMPL import batch_rodrigues


def render_mesh(vertices, faces, position, focal_point, up, scale=1.2,
                size=(400, 400), color='lightblue',
                ground_y=None, ground_center=None, ground_size=4.0):
    plotter = pv.Plotter(off_screen=True, window_size=size)

    # Ground plane, drawn first so the body occludes it correctly.
    # Camera coordinates have y pointing down, so the floor sits at max-y.
    if ground_y is not None:
        gc = ground_center if ground_center is not None else focal_point
        plane = pv.Plane(center=(gc[0], ground_y, gc[2]), direction=(0, -1, 0),
                         i_size=ground_size, j_size=ground_size,
                         i_resolution=1, j_resolution=1)
        plotter.add_mesh(plane, color='gainsboro', opacity=1.0,
                         smooth_shading=False)
        grid = pv.Plane(center=(gc[0], ground_y - 1e-3, gc[2]),
                        direction=(0, -1, 0),
                        i_size=ground_size, j_size=ground_size,
                        i_resolution=int(ground_size * 2),
                        j_resolution=int(ground_size * 2))
        plotter.add_mesh(grid, style='wireframe', color='darkgray',
                         line_width=1)

    pv_faces = np.concatenate(
        (np.full((faces.shape[0], 1), 3), faces), axis=1
    ).ravel()
    mesh = pv.PolyData(vertices, pv_faces)
    plotter.add_mesh(mesh, color=color, smooth_shading=True)
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


def render_raw_events_3d(event_list, fid, img_size=256, window=6,
                         max_events_per_frame=6000, time_scale=1.6,
                         size=(400, 400)):
    """Render a sliding window of raw events as a jAER-style 3D space-time cloud.

    Events from the last `window` frames form a 3D point cloud with time as
    a real third axis: the newest events enter at the front face and older
    events recede along the time axis as the video progresses. Each event is
    an individual point at its exact (x, y, t); positive polarity = blue,
    negative = red. Viewed from an oblique perspective so all three
    dimensions are visible.

    Args:
        event_list: list of per-frame [N, 4] tensors/arrays of (x, y, t, p)
        fid: current frame index (window covers frames [fid-window+1, fid])
        img_size: spatial resolution for normalization
        window: number of past frames in the space-time volume
        max_events_per_frame: subsample cap per frame
        time_scale: length of the time axis relative to the spatial axes
        size: output image size
    Returns:
        img: [H, W, 3] uint8 BGR image
    """
    start = max(0, fid - window + 1)

    pts_list = []
    col_list = []
    for slot, f_idx in enumerate(range(start, fid + 1)):
        ev = event_list[f_idx]
        if isinstance(ev, torch.Tensor):
            ev = ev.detach().cpu().numpy()
        if ev.shape[0] == 0:
            continue
        if ev.shape[0] > max_events_per_frame:
            idx = np.random.choice(ev.shape[0], max_events_per_frame, replace=False)
            idx.sort()
            ev = ev[idx]

        # Mirror x so the event cloud matches the mesh views' left/right
        x = 1.0 - ev[:, 0] / img_size
        y = ev[:, 1] / img_size

        # Continuous time within the window: frame slot + relative time inside frame
        t = ev[:, 2]
        t_min, t_max = t.min(), t.max()
        if t_max - t_min > 1e-6:
            t_rel = (t - t_min) / (t_max - t_min)
        else:
            t_rel = np.zeros_like(t)
        # Newest frame at the front (t=0), older frames recede
        age = (fid - f_idx) - t_rel + 1.0  # in (0, window]
        t_axis = age / window * time_scale

        pts_list.append(np.stack([x, t_axis, y], axis=1))

        colors = np.zeros((len(ev), 3), dtype=np.uint8)
        pos = ev[:, 3] > 0
        colors[pos] = [50, 80, 255]   # blue
        colors[~pos] = [255, 50, 50]  # red
        col_list.append(colors)

    if not pts_list:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)  # size is (W, H)

    pts = np.concatenate(pts_list, axis=0)
    colors = np.concatenate(col_list, axis=0)

    plotter = pv.Plotter(off_screen=True, window_size=size)
    cloud = pv.PolyData(pts)
    cloud['colors'] = colors
    plotter.add_points(cloud, scalars='colors', rgb=True, point_size=1.5,
                       render_points_as_spheres=False)

    # Wireframe box marking the space-time volume
    box = pv.Box(bounds=(0, 1, 0, time_scale, 0, 1))
    plotter.add_mesh(box, style='wireframe', color='gray', line_width=1)

    # Oblique perspective view: x to the right, time receding into the
    # screen, image y downward
    plotter.camera_position = [
        (2.0, -2.5, -0.9),
        (0.5, time_scale * 0.4, 0.5),
        (0, 0, -1),
    ]
    plotter.camera.view_angle = 30
    plotter.set_background('black')
    plotter.add_text('t', position=(int(size[0] * 0.68), int(size[1] * 0.16)),
                     font_size=11, color='gray')

    img = plotter.screenshot(return_img=True)
    plotter.close()

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def write_video(path, frames, fps):
    """Write frames to an H.264 mp4 that plays in browsers.

    OpenCV writes MPEG-4 Part 2 ('mp4v'), which most browsers cannot decode,
    so the result is re-encoded with ffmpeg when it is available.
    """
    import shutil
    import subprocess

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    # Some environments (e.g. conda) ship a stripped ffmpeg without libx264,
    # so pick the first binary that can actually encode H.264.
    ffmpeg = None
    for cand in ['/usr/bin/ffmpeg', shutil.which('ffmpeg')]:
        if not cand or not os.path.exists(cand):
            continue
        probe = subprocess.run([cand, '-hide_banner', '-encoders'],
                               capture_output=True, text=True)
        if 'libx264' in probe.stdout:
            ffmpeg = cand
            break
    if ffmpeg is None:
        print(f'  [warn] no ffmpeg with libx264; {os.path.basename(path)} '
              f'stays MPEG-4 and may not play in browsers')
        return

    tmp = path + '.h264.mp4'
    res = subprocess.run(
        [ffmpeg, '-y', '-loglevel', 'error', '-i', path,
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
         '-movflags', '+faststart', tmp],
        capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(tmp):
        os.replace(tmp, path)
    else:
        print(f'  [warn] H.264 transcode failed: {res.stderr.strip()[:200]}')
        if os.path.exists(tmp):
            os.remove(tmp)


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
    parser.add_argument('--frames_per_clip', type=int, default=32,
                        help='continuous-time queries decoded per clip')
    parser.add_argument('--show_gt', action='store_true', default=True,
                        help='render a ground-truth mesh panel')
    parser.add_argument('--resolution', type=int, default=400)
    parser.add_argument('--skip', type=int, default=8)
    parser.add_argument('--our_data', action='store_true', default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--contrast_loss', type=float, default=0)
    parser.add_argument('--test_high_fps', action='store_true')
    parser.add_argument('--target_action', type=str, default=None)

    my_args = parser.parse_args()
    for k, v in vars(my_args).items():
        setattr(args, k, v)

    from event_hpe.final_config import apply_final_config
    apply_final_config(args)

    # The 3D event visualization needs the raw event stream
    args.raw_events = True

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

        # Consecutive, non-overlapping clips so the stitched motion is
        # temporally continuous. Clips are indexed one source frame apart,
        # and each clip spans clip_len * skip source frames.
        span = args.clip_len * args.skip
        selected = clips[::span][:args.num_clips]

        print(f'\n=== {action}: {len(selected)} clips ===')

        action_verts = []
        action_gt_verts = []
        action_events = []
        n_out = args.frames_per_clip

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

            L = data['events'].shape[0]

            # Query the continuous-time motion field densely within the clip
            model.resize_smpl(n_out + 1)
            decode_ts = torch.linspace(-1, 1, n_out).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(batch, test=True,
                            override_clip_length=n_out, decode_ts=decode_ts)

            verts = out['verts'].detach().cpu().numpy()[0]  # [n_out+1, V, 3]
            if faces is None:
                faces = model.smpl.faces

            # Ground-truth meshes at the labelled keyframes
            gt_verts = None
            if args.show_gt:
                init_shape = batch['init_shape']
                if init_shape.dim() == 3:
                    init_shape = init_shape[:, 0]
                gt_theta = batch['theta'][0]                 # [L, 72]
                gt_tran = batch['tran'][0, :, 0, :]          # [L, 3]
                gt_rotmats = batch_rodrigues(
                    gt_theta.reshape(-1, 3)).view(L, 24, 3, 3)
                model.resize_smpl(L)
                with torch.no_grad():
                    gv, _, _ = model.smpl(
                        beta=init_shape[:, 75:85].repeat(L, 1),
                        rotmats=gt_rotmats, get_skin=True)
                gt_verts = (gv + gt_tran[:, None, :]).detach().cpu().numpy()

            # Split this clip's raw events into n_out equal-duration bins so
            # the event stream advances at the same rate as the mesh
            raw_ev = data.get('raw_events', None)
            bins = [torch.zeros(0, 4) for _ in range(n_out)]
            if raw_ev is not None and raw_ev.shape[0] > 0:
                t = raw_ev[:, 2]
                t0, t1 = t.min().item(), t.max().item()
                if t1 - t0 > 1e-9:
                    slot = ((t - t0) / (t1 - t0) * n_out).long().clamp(0, n_out - 1)
                    for j in range(n_out):
                        bins[j] = raw_ev[slot == j]

            # Skip the init frame (t=0), keep the decoded frames
            for j in range(n_out):
                action_verts.append(verts[j + 1])
                action_events.append(bins[j])
                if gt_verts is not None:
                    gt_idx = min(int(j / n_out * L), L - 1)
                    action_gt_verts.append(gt_verts[gt_idx])

        if not action_verts:
            continue

        all_action_frames[action] = (action_verts, action_events, action_gt_verts)
        print(f'  {len(action_verts)} total frames '
              f'({len(selected)} clips x {n_out} decoded)')

    # Render all actions
    res = args.resolution
    rendered_by_action = {}
    for action, (verts_list, event_list, gt_list) in all_action_frames.items():
        print(f'\nRendering {action}: {len(verts_list)} frames...')

        # Fixed camera for the whole action so global translation stays
        # visible, framed to contain every frame of both prediction and GT.
        all_v = np.concatenate([np.stack(verts_list)] +
                               ([np.stack(gt_list)] if gt_list else []), axis=0)
        lo = all_v.reshape(-1, 3).min(axis=0)
        hi = all_v.reshape(-1, 3).max(axis=0)
        center = (lo + hi) / 2.0

        # Floor height: median lowest vertex (max y, since y points down).
        # Taken from GT when available.
        floor_src = gt_list if gt_list else verts_list
        ground_y = float(np.median([fv[:, 1].max() for fv in floor_src]))

        # Include the floor in the vertical extent so it stays in frame
        span_y = max(hi[1], ground_y) - lo[1]
        margin = 1.25
        scale_front = max(hi[0] - lo[0], span_y) / 2.0 * margin
        scale_side = max(hi[2] - lo[2], span_y) / 2.0 * margin
        cam_y = (min(lo[1], ground_y) + max(hi[1], ground_y)) / 2.0
        d = 5.0
        # Lift the camera slightly above the focal point (y points down) so the
        # floor is seen at an angle instead of edge-on
        cam_lift = 0.9

        rendered = []
        for fid in tqdm(range(len(verts_list))):
            v = verts_list[fid]

            front = render_mesh(v, faces,
                                [center[0], cam_y - cam_lift, center[2] - d],
                                [center[0], cam_y, center[2]],
                                [0, -1, 0], scale=scale_front, size=(res, res),
                                ground_y=ground_y, ground_center=center)
            side = render_mesh(v, faces,
                               [center[0] + d, cam_y - cam_lift, center[2]],
                               [center[0], cam_y, center[2]],
                               [0, -1, 0], scale=scale_side, size=(res, res),
                               ground_y=ground_y, ground_center=center)

            # Sliding space-time window of raw events ending at this frame
            ev_img = render_raw_events_3d(event_list, fid, img_size=args.img_size, size=(res, res))
            front_bgr = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
            cv2.putText(front_bgr, 'Ours (front)', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            side_bgr = cv2.cvtColor(side, cv2.COLOR_RGB2BGR)
            cv2.putText(side_bgr, 'Ours (side)', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Add action label
            cv2.putText(front_bgr, action.split('_')[0], (10, res - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            panels = [ev_img, front_bgr, side_bgr]

            if gt_list:
                gv = gt_list[fid]
                # Same fixed camera as the prediction panels for direct comparison
                gt_img = render_mesh(gv, faces,
                                     [center[0], cam_y - cam_lift, center[2] - d],
                                     [center[0], cam_y, center[2]],
                                     [0, -1, 0], scale=scale_front, size=(res, res),
                                     color='wheat',
                                     ground_y=ground_y, ground_center=center)
                gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
                cv2.putText(gt_bgr, 'GT (15 FPS)', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                panels.append(gt_bgr)

            frame = np.concatenate(panels, axis=1)
            rendered.append(frame)

        rendered_by_action[action] = rendered

        video_path = os.path.join(args.save_folder, f'{action}.mp4')
        write_video(video_path, rendered, args.fps)
        print(f'  Saved: {video_path} ({len(rendered)} frames, {len(rendered)/args.fps:.1f}s)')

    # Combine all actions into one compilation video
    print('\nCreating compilation video...')
    all_frames = []
    for action in sorted(rendered_by_action.keys()):
        all_frames.extend(rendered_by_action[action])

    comp_path = os.path.join(args.save_folder, 'compilation.mp4')
    write_video(comp_path, all_frames, args.fps)
    print(f'Saved compilation: {comp_path} ({len(all_frames)} frames, {len(all_frames)/args.fps:.1f}s)')


if __name__ == '__main__':
    main()
