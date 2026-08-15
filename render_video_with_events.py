"""Render video with events + multi-view SMPL mesh.

Runs inference on CPU to avoid GPU conflicts, renders events alongside
the predicted mesh from multiple views.

Usage:
    PYTHONPATH=NeMF/src:$PYTHONPATH python render_video_with_events.py \
        --model_dir outputs/gmp_stage3_v5/.../model_events_pose.pkl \
        --fps 30 --sample_idx 200
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


def event_image_to_rgb(event_vol, idx):
    """Convert an 8-channel event volume slice to an RGB visualization."""
    ev = event_vol[idx].detach().cpu().numpy() if isinstance(event_vol, torch.Tensor) else event_vol[idx]
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


def main():
    from NeMF.src.arguments import Arguments
    args = Arguments('NeMF/configs', filename='application.yaml')
    args.config_path = 'NeMF/configs'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='data/mmhpsd')
    parser.add_argument('--model_dir', type=str,
                        default='ckpts/mmhpsd/model_events_pose.pkl')
    parser.add_argument('--event_folder', type=str,
                        default='data/mmhpsd_events/')
    parser.add_argument('--save_folder', type=str, default='outputs/videos/')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--sample_idx', type=int, default=300)
    parser.add_argument('--resolution', type=int, default=400)
    parser.add_argument('--our_data', action='store_true')
    parser.add_argument('--skip', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--contrast_loss', type=float, default=0)
    parser.add_argument('--test_high_fps', action='store_true')
    parser.add_argument('--target_action', type=str, default=None)

    my_args = parser.parse_args()
    for k, v in vars(my_args).items():
        setattr(args, k, v)

    from event_hpe.final_config import apply_final_config
    apply_final_config(args)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    from pl_data.event_hpe_dm import EventHPEDataModule
    dm = EventHPEDataModule(args.data_root, batch_size=1, num_workers=0,
                            target_action=args.target_action, args=args)
    dataset = dm.val_db
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    from pl_models.event_human_model import EventHumanModel
    args.batch_size = 1
    model = EventHumanModel(args)

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

    for idx, data in enumerate(loader):
        if idx < args.sample_idx:
            continue
        if idx > args.sample_idx:
            break

        for k in data:
            if k != 'info':
                data[k] = data[k].to(device, dtype=dtype)

        action = data['info'][0][0]
        start_frame = data['info'][1][0][0].item()
        print(f'Sample {idx}: {action}, start_frame={start_frame}')

        events = data['events'][0]  # [T, C, H, W]
        L = events.shape[0]

        # Decode at high temporal resolution
        num_output = max(args.fps, L + 1)
        model.resize_smpl(num_output + 1)
        decode_ts = torch.linspace(-1, 1, num_output).unsqueeze(0)

        print(f'Running inference ({num_output} frames on CPU)...')
        with torch.no_grad():
            out = model(data, test=True, override_clip_length=num_output, decode_ts=decode_ts)

        verts = out['verts'].detach().cpu().numpy()[0]  # [T+1, V, 3]
        faces = model.smpl.faces
        N = verts.shape[0]
        print(f'Got {N} mesh frames')

        # Build event images (one per original timestep, repeated for high fps)
        event_imgs = []
        for t in range(L):
            event_imgs.append(event_image_to_rgb(events, t))

        res = args.resolution
        print(f'Rendering {N} frames at {res}x{res} per view...')
        rendered = []
        for fid in tqdm(range(N)):
            c = verts[fid].mean(axis=0)
            d = 5.0

            front = render_mesh(verts[fid], faces,
                                [c[0], c[1], c[2] - d], [c[0], c[1], c[2]],
                                [0, -1, 0], scale=1.2, size=(res, res))
            side = render_mesh(verts[fid], faces,
                               [c[0] + d, c[1], c[2]], [c[0], c[1], c[2]],
                               [0, -1, 0], scale=1.2, size=(res, res))

            # Map high-fps frame index to event image index
            ev_idx = min(int(fid / N * L), L - 1)
            ev_img = cv2.resize(event_imgs[ev_idx], (res, res))

            # Add labels
            cv2.putText(ev_img, 'Events', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            front_bgr = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
            cv2.putText(front_bgr, 'Front', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            side_bgr = cv2.cvtColor(side, cv2.COLOR_RGB2BGR)
            cv2.putText(side_bgr, 'Side', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            frame = np.concatenate([ev_img, front_bgr, side_bgr], axis=1)
            rendered.append(frame)

        video_path = os.path.join(args.save_folder, f'{action}_{start_frame}.mp4')
        h, w = rendered[0].shape[:2]
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
        for f in rendered:
            writer.write(f)
        writer.release()
        print(f'Saved: {video_path} ({N} frames, {N/args.fps:.1f}s)')
        break


if __name__ == '__main__':
    main()
