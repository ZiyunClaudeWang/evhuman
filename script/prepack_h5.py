"""Pre-pack event volumes and poses into per-action HDF5 files for fast loading.

Instead of loading 16 individual PNGs + pickle files per sample,
this packs everything into one HDF5 file per action sequence.
The data loader can then read contiguous chunks with a single I/O call.

Usage:
    python script/prepack_h5.py \
        --data_root data/beahm \
        --output_dir data/beahm_h5 \
        --img_size 256
"""
import argparse
import os
import pickle

import cv2
import h5py
import joblib
import numpy as np
from tqdm import tqdm


def pack_action(data_root, action, output_dir, img_size):
    """Pack all event volumes and poses for one action into an HDF5 file."""
    pose_dir = os.path.join(data_root, 'pose_events', action)
    event_dir = os.path.join(data_root, f'events_{img_size}', action)

    if not os.path.isdir(pose_dir) or not os.path.isdir(event_dir):
        print(f'  Skipping {action}: missing directories')
        return

    pose_info_path = os.path.join(pose_dir, 'pose_info.pkl')
    if not os.path.exists(pose_info_path):
        print(f'  Skipping {action}: no pose_info.pkl')
        return

    frame_indices = joblib.load(pose_info_path)

    # Count available frames
    all_events = []
    all_theta = []
    all_tran = []
    all_joints3d = []
    all_joints2d = []
    all_intri = []
    all_frame_idx = []

    for idx in tqdm(frame_indices, desc=action, leave=False):
        pose_path = os.path.join(pose_dir, f'pose{idx:04d}.pkl')
        event_path = os.path.join(event_dir, f'event{idx:04d}.png')

        if not os.path.exists(pose_path) or not os.path.exists(event_path):
            continue

        pose_data = joblib.load(pose_path)
        if len(pose_data) == 6:
            beta, theta, tran, joints3d, joints2d, intri = pose_data
        elif len(pose_data) == 5:
            beta, theta, tran, joints3d, joints2d = pose_data
            intri = None
        else:
            continue

        event_img = cv2.imread(event_path, -1)
        if event_img is None:
            continue

        all_events.append(event_img)
        all_theta.append(theta)
        all_tran.append(tran)
        all_joints3d.append(joints3d)
        all_joints2d.append(joints2d)
        if intri is not None:
            all_intri.append(intri)
        all_frame_idx.append(idx)

    if not all_events:
        print(f'  Skipping {action}: no valid frames')
        return

    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, f'{action}.h5')

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('events', data=np.stack(all_events), compression='lzf')
        f.create_dataset('theta', data=np.stack(all_theta))
        f.create_dataset('tran', data=np.stack(all_tran))
        f.create_dataset('joints3d', data=np.stack(all_joints3d))
        f.create_dataset('joints2d', data=np.stack(all_joints2d))
        f.create_dataset('frame_indices', data=np.array(all_frame_idx))
        if all_intri:
            f.create_dataset('intri', data=np.stack(all_intri))
        f.attrs['beta'] = beta  # same for all frames in a sequence
        f.attrs['action'] = action
        f.attrs['img_size'] = img_size
        f.attrs['num_frames'] = len(all_events)

    print(f'  {action}: {len(all_events)} frames -> {h5_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=256)
    args = parser.parse_args()

    pose_events_dir = os.path.join(args.data_root, 'pose_events')
    actions = sorted(os.listdir(pose_events_dir))
    print(f'Found {len(actions)} actions')

    for action in actions:
        pack_action(args.data_root, action, args.output_dir, args.img_size)

    print('Done!')


if __name__ == '__main__':
    main()
