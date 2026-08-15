"""Fast H5-based data loader for BEAHM dataset.

Reads pre-packed HDF5 files (one per action) instead of individual PNGs
and pickle files. ~5-10x faster data loading.

Usage: set h5_dir in constructor to the directory with per-action .h5 files
created by script/prepack_h5.py.
"""
import os
import pickle

import cv2
import h5py
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset


class OursTrackingDataloaderH5(Dataset):
    def __init__(
            self,
            data_dir='data/beahm',
            h5_dir=None,
            max_steps=16,
            num_steps=8,
            skip=2,
            events_input_channel=8,
            img_size=256,
            mode='train',
            use_flow=True,
            use_hmr_feats=False,
            target_action=None,
            raw_events=False,
            event_folder="data/beahm_events/",
            use_volumes=False,
            test_high_fps=False,
    ):
        self.data_dir = data_dir
        self.h5_dir = h5_dir or (data_dir + '_h5')
        self.events_input_channel = events_input_channel
        self.skip = skip
        self.max_steps = max_steps
        self.num_steps = num_steps
        self.img_size = img_size
        self.scale = self.img_size / 640.
        self.use_hmr_feats = use_hmr_feats
        self.use_flow = use_flow
        self.raw_events = raw_events
        self.event_folder = event_folder
        self.use_volumes = use_volumes
        self.test_high_fps = test_high_fps
        self.mode = mode

        # Load H5 files into memory-mapped handles
        self.h5_data = {}
        self._load_h5_files()

        # Build clip list
        cache_path = '%s/%s_track%02i%02i.pkl' % (self.data_dir, self.mode, self.num_steps, self.skip)
        if os.path.exists(cache_path):
            self.all_clips = pickle.load(open(cache_path, 'rb'))
        else:
            self.all_clips = self._obtain_all_clips()

        if target_action:
            self.all_clips = [(a, f) for a, f in self.all_clips if target_action in a]

        print('[%s H5] %i clips, track%02i%02i.pkl' % (self.mode, len(self.all_clips), self.num_steps, self.skip))

    def _load_h5_files(self):
        """Load all H5 files and build frame index lookup."""
        self.h5_handles = {}
        self.frame_to_h5idx = {}

        for fname in sorted(os.listdir(self.h5_dir)):
            if not fname.endswith('.h5'):
                continue
            action = fname[:-3]
            path = os.path.join(self.h5_dir, fname)
            h5f = h5py.File(path, 'r')
            self.h5_handles[action] = h5f

            frame_indices = h5f['frame_indices'][:]
            idx_map = {int(fi): i for i, fi in enumerate(frame_indices)}
            self.frame_to_h5idx[action] = idx_map

    def _obtain_all_clips(self):
        """Build clip list from H5 data (same split logic as original loader)."""
        all_clips = []
        test_motions = ['Squat', 'Starjump', 'Walking', 'Jogging',
                        'Jumpforback', 'Jumpsideway', 'Jumpupdown',
                        'Leanleft', 'Leanright']

        for action in sorted(self.h5_handles.keys()):
            motion = action.split('_')[0]
            subject = action.split('_')[1]
            idx = action.split('_')[-1]
            # public release anonymizes ziyan -> n2 (indices preserved)
            is_test = (motion in test_motions and subject in ['ziyan', 'n2'] and idx in ['1'])

            if (self.mode == 'test') != is_test:
                continue

            h5f = self.h5_handles[action]
            frame_indices = h5f['frame_indices'][:]

            for i in range(len(frame_indices) - self.max_steps * self.skip):
                frame_idx = int(frame_indices[i])
                end_frame_idx = frame_idx + self.max_steps * self.skip

                # Check all required frames exist
                idx_map = self.frame_to_h5idx[action]
                required = [frame_idx + self.skip * k for k in range(self.max_steps + 1)]
                if all(r in idx_map for r in required):
                    all_clips.append((action, frame_idx))

        # Cache for next time
        cache_path = '%s/%s_track%02i%02i.pkl' % (self.data_dir, self.mode, self.num_steps, self.skip)
        with open(cache_path, 'wb') as f:
            pickle.dump(all_clips, f)

        return all_clips

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):
        action, frame_idx = self.all_clips[idx]
        h5f = self.h5_handles[action]
        idx_map = self.frame_to_h5idx[action]

        if self.mode == 'train':
            next_frames_idx = self.skip * np.sort(np.random.choice(
                np.arange(1, self.max_steps + 1), self.num_steps, replace=False))
        else:
            next_frames_idx = self.skip * np.arange(1, self.num_steps + 1)

        sample_frames_idx = np.append(frame_idx, frame_idx + next_frames_idx)

        # Load init shape from first frame
        h5_idx_0 = idx_map[frame_idx]
        beta = h5f.attrs['beta']
        theta_0 = h5f['theta'][h5_idx_0]
        tran_0 = h5f['tran'][h5_idx_0]
        init_shape = np.concatenate([tran_0.reshape(1, -1), theta_0.reshape(1, -1),
                                      beta.reshape(1, -1)], axis=1)

        if self.use_hmr_feats:
            hmr_path = '%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx)
            if os.path.exists(hmr_path):
                _, _, _, _, hmr_feats = joblib.load(hmr_path)
            else:
                hmr_feats = np.zeros([2048])
        else:
            hmr_feats = np.zeros([2048])

        events_list = []
        flows_list = []
        theta_list, tran_list, joints2d_list, joints3d_list = [], [], [], []

        high_fps_theta_list = []
        high_fps_tran_list = []
        high_fps_joints2d_list = []
        high_fps_joints3d_list = []

        for i in range(self.num_steps):
            start_idx = sample_frames_idx[i]
            end_idx = sample_frames_idx[i + 1]

            # Read event volumes for this step (contiguous H5 reads)
            single_events_frame = []
            for j in range(start_idx, end_idx):
                if j in idx_map:
                    ev = h5f['events'][idx_map[j]]  # single H5 read
                    single_events_frame.append(ev)
                else:
                    single_events_frame.append(np.zeros((self.img_size, self.img_size, 4), dtype=np.float32))

            single_events_frame = np.concatenate(single_events_frame, axis=2).astype(np.float32)

            # Aggregate to target channels
            if single_events_frame.shape[2] > self.events_input_channel:
                skip_c = single_events_frame.shape[2] // self.events_input_channel
                idx1 = skip_c * np.arange(self.events_input_channel)
                idx2 = idx1 + skip_c
                idx2[-1] = max(idx2[-1], single_events_frame.shape[2])
                single_events_frame = np.stack(
                    [(np.sum(single_events_frame[:, :, c1:c2], axis=2) > 0)
                     for (c1, c2) in zip(idx1, idx2)], axis=2)

            events_list.append(single_events_frame)
            flows_list.append(np.zeros([2, self.img_size, self.img_size]))

            # Load pose at end_idx
            if end_idx in idx_map:
                h5_i = idx_map[end_idx]
                theta_list.append(h5f['theta'][h5_i])
                tran_list.append(h5f['tran'][h5_i])
                joints2d_list.append(h5f['joints2d'][h5_i])
                joints3d_list.append(h5f['joints3d'][h5_i])
            else:
                theta_list.append(np.zeros(72))
                tran_list.append(np.zeros((1, 3)))
                joints2d_list.append(np.zeros((24, 2)))
                joints3d_list.append(np.zeros((24, 3)))

            # High FPS data
            if self.test_high_fps:
                for idd in range(start_idx + 1, end_idx + 1):
                    if idd in idx_map:
                        h5_i = idx_map[idd]
                        high_fps_theta_list.append(h5f['theta'][h5_i])
                        high_fps_tran_list.append(h5f['tran'][h5_i])
                        high_fps_joints2d_list.append(h5f['joints2d'][h5_i])
                        high_fps_joints3d_list.append(h5f['joints3d'][h5_i])

        events_array = np.stack(events_list, axis=0)
        events_array = np.transpose(events_array, (0, 3, 1, 2))

        theta_array = np.stack(theta_list, axis=0)
        tran_array = np.stack(tran_list, axis=0)
        joints2d_array = np.stack(joints2d_list, axis=0)
        joints3d_array = np.stack(joints3d_list, axis=0)

        # Camera intrinsics
        if 'intri' in h5f:
            intri = h5f['intri'][h5_idx_0]
        else:
            intri = np.zeros(4)

        one_sample = {
            'events': torch.from_numpy(events_array).float(),
            'flows': torch.from_numpy(np.stack(flows_list, axis=0)).float(),
            'init_shape': torch.from_numpy(init_shape).float(),
            'hidden_feats': torch.from_numpy(hmr_feats).float(),
            'theta': torch.from_numpy(theta_array).float(),
            'tran': torch.from_numpy(tran_array[:, None, :]).float(),
            'joints2d': torch.from_numpy(joints2d_array).float(),
            'joints3d': torch.from_numpy(joints3d_array).float(),
            'info': (action, torch.from_numpy(sample_frames_idx)),
            'intri': torch.from_numpy(intri * self.scale).float(),
        }

        if self.test_high_fps:
            one_sample['high_fps_theta'] = torch.from_numpy(np.stack(high_fps_theta_list)).float()
            one_sample['high_fps_tran'] = torch.from_numpy(np.stack(high_fps_tran_list)[:, None, :]).float()
            one_sample['high_fps_joints2d'] = torch.from_numpy(np.stack(high_fps_joints2d_list)).float()
            one_sample['high_fps_joints3d'] = torch.from_numpy(np.stack(high_fps_joints3d_list)).float()

        return one_sample

    def __del__(self):
        for h5f in self.h5_handles.values():
            h5f.close()
