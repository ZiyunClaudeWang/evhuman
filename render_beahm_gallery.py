"""Render BEAHM dataset showcase clips: raw events, an events+GT overlay,
and the 120 FPS ground-truth mesh — all in the event camera's own view.

Each released H5 is self-contained: raw events (x, y, t, p), annotations
at exactly 120 FPS (SMPL pose/shape, root R/T), and calibration. Geometry
(from the capture pipeline, EventMonoPose):

- Annotations are EasyMocap multi-view fits in the kalibr cam0 (FLIR) world
  frame; calibration/event_extr maps that frame into the event camera.
- The event calibration lives in the transposed (portrait) pixel frame:
  (u, v) = (H5 y, H5 x). Displaying events as image[x, y] is already the
  calibrated orientation — no flip is involved.
- Projection uses the full MEI omnidirectional model
  (cv2.omnidir.projectPoints with event_intr/event_dist), so the overlay is
  exact including lens distortion.

Usage:
    PYTHONPATH=NeMF/src:$PYTHONPATH python render_beahm_gallery.py \
        --h5_dir data/beahm_h5/extreme --save_folder outputs/beahm_gallery
"""
import argparse
import math
import os

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from event_hpe.SMPL import SMPL, batch_rodrigues
from render_demo_video import write_video

RAW_W, RAW_H = 480, 640          # portrait event sensor


def render_mesh_event_view(vertices, faces, K):
    """Render the mesh from the event camera: perspective from the origin,
    y down, in the transposed portrait frame, using the release pipeline's
    rectified-perspective intrinsics (0.6 * focal, same principal point).
    Vertices must already be in event-camera coordinates."""
    import pyvista as pv
    fx, fy, cx, cy = K
    plotter = pv.Plotter(off_screen=True, window_size=(RAW_W, RAW_H))
    pv_faces = np.concatenate(
        (np.full((faces.shape[0], 1), 3), faces), axis=1).ravel()
    plotter.add_mesh(pv.PolyData(vertices, pv_faces), color='wheat',
                     smooth_shading=True)
    cam = plotter.camera
    plotter.camera_position = [(0, 0, 0), (0, 0, 5), (0, -1, 0)]
    cam.view_angle = math.degrees(2 * math.atan((RAW_H / 2) / fy))
    cam.SetWindowCenter(-(2 * cx - RAW_W) / RAW_W, (2 * cy - RAW_H) / RAW_H)
    cam.clipping_range = (0.5, 60)
    plotter.set_background('white')
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_dir', type=str, default='data/beahm_h5/extreme')
    parser.add_argument('--sequences', type=str, nargs='+', default=[
        'Taekwondo_n1_1.h5', 'Jumptwist_n2_1.h5',
        'Volleyball_n1_1.h5', 'Tennisswing_n2_2.h5'])
    parser.add_argument('--save_folder', type=str, default='outputs/beahm_gallery')
    parser.add_argument('--smpl_dir', type=str,
                        default='event_hpe/smpl_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    parser.add_argument('--n_frames', type=int, default=180)
    parser.add_argument('--fps', type=int, default=30,
                        help='playback fps (120 FPS source -> 4x slow motion)')
    parser.add_argument('--resolution', type=int, default=380,
                        help='panel height; width follows the sensor aspect')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_folder, exist_ok=True)
    res_h = args.resolution
    res_w = int(res_h * RAW_W / RAW_H)

    for seq in args.sequences:
        path = os.path.join(args.h5_dir, seq)
        if not os.path.exists(path):
            print(f'[skip] {path} not found')
            continue
        name = seq.replace('.h5', '')
        print(f'=== {name} ===')

        with h5py.File(path, 'r') as f:
            xi, fu, fv, cu, cv_ = f['calibration/event_intr'][:].astype(np.float64)
            dist = f['calibration/event_dist'][:].astype(np.float64)
            extr = f['calibration/event_extr'][:].astype(np.float64)
            R_ann = f['annotations/R'][:]
            T = f['annotations/T'][:]
            poses = f['annotations/poses'][:]
            shape = f['annotations/shape'][:]
            annot_ts = f['events/event_annot_ts'][:]
            ev_t = f['events/t'][:]

            starts = np.searchsorted(ev_t, annot_ts)
            counts = np.diff(starts)
            n = min(args.n_frames, len(counts) - 1)
            w0 = int(np.argmax(np.convolve(counts, np.ones(n), 'valid'))) + 1
            print(f'  window: frames {w0}..{w0 + n} of {len(annot_ts)}')

            ev_x = f['events/x'][starts[w0 - 1]:starts[w0 + n - 1]]
            ev_y = f['events/y'][starts[w0 - 1]:starts[w0 + n - 1]]
            ev_p = f['events/p'][starts[w0 - 1]:starts[w0 + n - 1]]
            ev_tt = ev_t[starts[w0 - 1]:starts[w0 + n - 1]]

        # GT meshes at 120 FPS. EasyMocap storage convention: poses have a
        # zero root; R rotates the posed body about the SMPL-space ORIGIN
        # (not the root joint), then T translates:  V = R @ SMPL(pose) + T.
        smpl = SMPL(args.smpl_dir, n).to(device)
        faces = smpl.faces
        rotmats = batch_rodrigues(
            torch.tensor(poses[w0:w0 + n].reshape(-1, 3), dtype=torch.float32,
                         device=device)).view(n, 24, 3, 3)
        beta = torch.tensor(shape[w0:w0 + n], dtype=torch.float32, device=device)
        with torch.no_grad():
            v, _, _ = smpl(beta=beta, rotmats=rotmats, get_skin=True)
        v = v.cpu().numpy().astype(np.float64)
        Rh = np.stack([cv2.Rodrigues(r)[0] for r in R_ann[w0:w0 + n]])
        verts = np.einsum('nij,nvj->nvi', Rh, v) + T[w0:w0 + n][:, None, :]

        bounds = np.searchsorted(ev_tt, annot_ts[w0 - 1:w0 + n])

        # annotation world (cam0) -> event camera, then exact MEI projection
        Kmat = np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1]])
        D = dist.reshape(1, 4)
        R_cam, t_cam = extr[:3, :3], extr[:3, 3]
        verts_evt = verts @ R_cam.T + t_cam

        frames = []
        for i in tqdm(range(n)):
            s, e = bounds[i], bounds[i + 1]
            # events in the calibrated (transposed portrait) frame: image[x, y]
            ev_img = np.zeros((RAW_H, RAW_W, 3), np.uint8)
            pos = ev_p[s:e] > 0
            np.add.at(ev_img[:, :, 2], (ev_x[s:e][pos], ev_y[s:e][pos]), 120)
            np.add.at(ev_img[:, :, 0], (ev_x[s:e][~pos], ev_y[s:e][~pos]), 120)
            ev = cv2.resize(ev_img, (res_w, res_h), interpolation=cv2.INTER_AREA)
            ev = np.ascontiguousarray(ev)
            cv2.putText(ev, 'Events', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # overlay: exact MEI projection of the GT vertices
            ov = ev.copy()
            Ve = verts_evt[i][::6]
            pix, _ = cv2.omnidir.projectPoints(
                Ve.reshape(1, -1, 3), np.zeros(3), np.zeros(3), Kmat, xi, D)
            pu = pix[0, :, 0] * res_w / RAW_W
            pv_ = pix[0, :, 1] * res_h / RAW_H
            keep = (pu >= 0) & (pu < res_w) & (pv_ >= 0) & (pv_ < res_h)
            ov[pv_[keep].astype(int), pu[keep].astype(int)] = (0, 220, 0)
            cv2.putText(ov, 'Events + GT', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            mesh = render_mesh_event_view(verts_evt[i], faces,
                                          (0.6 * fu, 0.6 * fv, cu, cv_))
            mesh = cv2.cvtColor(mesh, cv2.COLOR_RGB2BGR)
            mesh = cv2.resize(mesh, (res_w, res_h), interpolation=cv2.INTER_AREA)
            mesh = np.ascontiguousarray(mesh)
            cv2.putText(mesh, 'GT 120 FPS', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(mesh, name.split('_')[0], (10, res_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            frames.append(np.concatenate([ev, ov, mesh], axis=1))

        out = os.path.join(args.save_folder, f'{name}.mp4')
        write_video(out, frames, args.fps)
        print(f'  Saved: {out} ({n} frames, {n / args.fps:.1f}s, '
              f'{120 / args.fps:.0f}x slow motion)')


if __name__ == '__main__':
    main()
