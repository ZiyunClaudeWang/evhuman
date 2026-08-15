"""Convert the released BEAHM H5 files into the processed layout that the
training and evaluation code consumes.

The public BEAHM release ships self-contained H5 files (raw events, 120 FPS
SMPL annotations, calibration). This script produces, per sequence:

    <out_dir>/events_256/<action>/eventNNNN.png   4-bin binary event frames,
                                                  undistorted to the release
                                                  pipeline's 256x256 view
    <out_dir>/pose_events/<action>/poseNNNN.pkl   (beta, theta, tran,
                                                  joints3d, joints2d, intri)
                                                  in the event-camera frame
    <out_dir>/pose_events/<action>/pose_info.pkl  frame index list

It is a faithful port of the capture pipeline's prepare_data.py: the same
undistortion (MEI model, 0.6 focal coefficient, 80 px pad, 256 resize), the
same conversion of the EasyMocap world-frame annotations into the event
camera (event_extr, with the origin-pivot to root-joint-pivot correction),
and the same intrinsics convention (0.24 * fu, 0.4 * (cu + 80), 0.4 * cv).

After preprocessing, point --data_root at <out_dir> and --event_folder at
the released H5 directory.

Usage:
    PYTHONPATH=NeMF/src:$PYTHONPATH python script/preprocess_beahm.py \
        --h5_dir data/beahm_h5/basic --out_dir data/beahm --num_workers 6
"""
import argparse
import glob
import os
import sys

import cv2
import h5py
import joblib
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from event_hpe.SMPL import SMPL, batch_rodrigues

IMAGE_SIZE = 256
NUM_PARTITIONS = 4
RAW_H, RAW_W = 640, 480          # transposed portrait event frame
FRAME_US = 8000                  # 120 FPS annotation spacing


def undistort_frame(frame, intr, dist):
    """MEI-model rectification of one portrait frame (release convention:
    P keeps the principal point and scales the focal by 0.6)."""
    xi = np.array([intr[0]])
    K = np.array([[intr[1], 0, intr[3]], [0, intr[2], intr[4]], [0, 0, 1.0]])
    P = np.array([[intr[1] * 0.6, 0, intr[3]],
                  [0, intr[2] * 0.6, intr[4]], [0, 0, 1.0]])
    map1, map2 = cv2.omnidir.initUndistortRectifyMap(
        K, dist, xi, np.eye(3), P, (RAW_W, RAW_H), cv2.CV_32F,
        cv2.omnidir.RECTIFY_PERSPECTIVE)
    return cv2.remap(frame, map1, map2, cv2.INTER_CUBIC)


def events_to_frame(events, intr, dist):
    """4 time-partition binary frames, undistorted, stacked to [H, W, 4]."""
    t0, t1 = events[0, 2], events[-1, 2]
    window = (t1 - t0) / NUM_PARTITIONS
    parts = []
    for i in range(NUM_PARTITIONS):
        frame = np.zeros([RAW_H, RAW_W])
        sel = ((events[:, 2] >= t0 + i * window) &
               (events[:, 2] < t0 + (i + 1) * window))
        frame[events[sel, 0], events[sel, 1]] = 1        # rows = H5 x
        parts.append(undistort_frame(frame, intr, dist))
    return np.stack(parts, axis=2)


def process_one(h5_path, out_dir, smpl_dir, device='cpu'):
    action = os.path.basename(h5_path)[:-3]
    os.makedirs(f'{out_dir}/events_{IMAGE_SIZE}/{action}', exist_ok=True)
    os.makedirs(f'{out_dir}/pose_events/{action}', exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        ev = np.stack([f['events/x'][:], f['events/y'][:],
                       f['events/t'][:], f['events/p'][:]], axis=1)
        trigger = f['events/event_annot_ts'][:]
        intr = f['calibration/event_intr'][:].astype(np.float64)
        dist = f['calibration/event_dist'][:].astype(np.float64)
        extr = f['calibration/event_extr'][:].astype(np.float64)
        R_ann = f['annotations/R'][:]
        T_ann = f['annotations/T'][:]
        poses = f['annotations/poses'][:]
        shape = f['annotations/shape'][:]

    n = len(trigger)
    R_cam, t_cam = extr[:3, :3], extr[:3, 3]

    # SMPL quantities (EasyMocap storage: zero-root poses; R rotates the
    # posed body about the SMPL-space origin, T translates)
    smpl = SMPL(smpl_dir, n).to(device)
    with torch.no_grad():
        rotmats = batch_rodrigues(
            torch.tensor(poses.reshape(-1, 3), dtype=torch.float32,
                         device=device)).view(n, 24, 3, 3)
        beta_t = torch.tensor(shape, dtype=torch.float32, device=device)
        # EasyMocap's return_smpl_joints treats the regressed rest joints
        # as pseudo-vertices and skins them with regressed blend weights and
        # regressed pose blendshapes; replicate that exactly
        from smplx.lbs import batch_rigid_transform
        J_std = smpl.J_regressor.t()                       # [24, 6890]
        v_shaped = (beta_t @ smpl.shapedirs).view(n, 6890, 3) + smpl.v_template
        j_rest = torch.einsum('jv,nvc->njc', J_std, v_shaped)
        w0 = smpl.weight[0]                                # [6890, 24]
        j_weights = J_std @ w0                             # [24, 24]
        pd = smpl.posedirs.view(207, 6890, 3)
        j_posedirs = torch.einsum('jv,pvc->pjc', J_std, pd).reshape(207, 72)
        eye3 = torch.eye(3, device=device)
        pose_feat = (rotmats[:, 1:] - eye3).reshape(n, 207)
        j_posed = j_rest + (pose_feat @ j_posedirs).view(n, 24, 3)
        _, A = batch_rigid_transform(rotmats, j_rest, smpl.parents,
                                     dtype=rotmats.dtype)
        T_j = torch.einsum('jk,nkab->njab', j_weights, A)
        j_local = (T_j[:, :, :3, :3] @ j_posed.unsqueeze(-1)).squeeze(-1) \
            + T_j[:, :, :3, 3]
        j0_t = j_rest[:, 0]
    j_local = j_local.cpu().numpy().astype(np.float64)
    j0 = j0_t.cpu().numpy().astype(np.float64)

    # per-frame intrinsics of the processed 256x256 view
    intri = (IMAGE_SIZE / 640.0) * np.array(
        [0.6 * intr[1], 0.6 * intr[2], intr[3] + 80, intr[4]])
    intr_c = np.array([[intri[0], 0., intri[2]],
                       [0., intri[1], intri[3]], [0., 0., 1.]])

    frame_indices = []
    events_count = []
    for idx in range(n):
        # events of this 1/120 s interval -> 4-bin undistorted frame
        t = trigger[idx]
        sel = ev[(ev[:, 2] > t) & (ev[:, 2] < t + FRAME_US)]
        if len(sel) == 0:
            sel = np.zeros((1, 4), dtype=ev.dtype)
        img = events_to_frame(sel.astype(np.int64), intr, dist)
        img = np.pad(img, ((0, 0), (80, 80), (0, 0)), 'constant')
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f'{out_dir}/events_{IMAGE_SIZE}/{action}/event%04d.png'
                    % idx, img.astype(np.uint8))
        frame_indices.append(int(idx))
        events_count.append(int(np.sum(img > 0)))

        # annotations -> event-camera frame, root-joint-pivot convention
        Rh = cv2.Rodrigues(R_ann[idx])[0]
        R_global = R_cam @ Rh
        theta = poses[idx].copy()
        theta[:3] = cv2.Rodrigues(R_global)[0].ravel()
        tran = (R_cam @ T_ann[idx] + t_cam
                + R_global @ j0[idx] - j0[idx])
        joints_world = (Rh @ j_local[idx].T).T + T_ann[idx]
        joints3d = (R_cam @ joints_world.T).T + t_cam
        pix = intr_c @ joints3d.T
        joints2d = (pix[:2] / pix[2]).T

        joblib.dump([shape[idx], theta, tran, joints3d.astype(np.float32),
                     joints2d, intri],
                    f'{out_dir}/pose_events/{action}/pose%04i.pkl' % idx)

    joblib.dump(frame_indices,
                f'{out_dir}/pose_events/{action}/pose_info.pkl')
    joblib.dump([frame_indices, events_count],
                f'{out_dir}/events_{IMAGE_SIZE}/{action}/{action}_info.pkl',
                compress=3)
    return action, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_dir', type=str, required=True,
                        help='released BEAHM H5 directory (e.g. basic/)')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--smpl_dir', type=str,
                        default='data/smpl/SMPL_NEUTRAL.pkl',
                        help='the smplx-packaged neutral model; the GT was '
                             'fit with it, so it must be used here (the '
                             'training SMPL stays basicModel v1.0.0)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sequences', type=str, nargs='*', default=None,
                        help='subset of file names; default: all *.h5')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.h5_dir, '*.h5')))
    if args.sequences:
        keep = set(args.sequences)
        files = [f for f in files if os.path.basename(f) in keep]
    print(f'{len(files)} sequences -> {args.out_dir}')

    if args.num_workers <= 1:
        for f in files:
            print('  %s: %d frames' % process_one(f, args.out_dir,
                                                  args.smpl_dir))
    else:
        import multiprocessing
        with multiprocessing.Pool(args.num_workers) as pool:
            jobs = [pool.apply_async(process_one,
                                     (f, args.out_dir, args.smpl_dir))
                    for f in files]
            for j in jobs:
                print('  %s: %d frames' % j.get())
    print('done')


if __name__ == '__main__':
    main()
