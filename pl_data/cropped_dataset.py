"""Dataset wrapper that applies detection-based crops to tracking samples.

Wraps OursTrackingDataloader / TrackingDataloader and yields the same dict,
with events/flows cropped to a canonical, subject-centred view. GT that
lives in image space (joints2d, intrinsics, raw event coordinates) is
remapped into the crop; 3D quantities (theta, tran, joints3d) are untouched
because the camera does not move — only the pixel sampling changes.

Box sources:
- 'gt': square box around the GT 2D joints (oracle; isolates the effect of
  cropping from detector noise)
- a path to a cache pkl produced by script/precompute_bboxes.py (detector)

The applied crop is returned as sample['crop_box'] = (cx, cy, size) along
with the original intrinsics in sample['intri_full'], so the weak-perspective
camera head can convert crop-frame predictions back to the full camera frame.
"""
import pickle

import numpy as np
import torch

from pl_data.crop_utils import (adjust_intrinsics, clip_union_bbox,
                                crop_resize_volume, remap_points_norm,
                                remap_raw_events)


class CroppedTrackingDataset(torch.utils.data.Dataset):

    def __init__(self, base_loader, img_size=256, out_size=256,
                 box_source='gt', pad_ratio=0.15, jitter=0.0):
        """
        Args:
            base_loader: OursTrackingDataloader or TrackingDataloader
            box_source: 'gt' or path to a bbox cache pkl
            pad_ratio: padding around the union box
            jitter: random scale/centre jitter (train-time augmentation),
                e.g. 0.05 shifts/scales the box by up to 5%
        """
        self.base = base_loader
        self.img_size = img_size
        self.out_size = out_size
        self.pad_ratio = pad_ratio
        self.jitter = jitter

        self.box_cache = None
        if box_source != 'gt':
            with open(box_source, 'rb') as f:
                self.box_cache = pickle.load(f)
            print(f'[CroppedTrackingDataset] {len(self.box_cache)} cached boxes '
                  f'from {box_source}')

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        # transparent passthrough for loader attributes (all_clips, mode, ...)
        return getattr(self.base, name)

    def _frame_boxes(self, sample, idx):
        """Per-frame (cx, cy, w, h) boxes, normalized."""
        if self.box_cache is not None:
            action, frame_idx = self.base.all_clips[idx]
            key = (action, int(frame_idx))
            if key in self.box_cache:
                return torch.as_tensor(self.box_cache[key], dtype=torch.float32)
        # oracle: tight box around GT 2D joints (already normalized [0,1])
        j2d = sample['joints2d']                     # [T, 24, 2]
        x0 = j2d[..., 0].min(dim=1).values
        x1 = j2d[..., 0].max(dim=1).values
        y0 = j2d[..., 1].min(dim=1).values
        y1 = j2d[..., 1].max(dim=1).values
        size = torch.max(x1 - x0, y1 - y0)
        return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, size, size], dim=1)

    def __getitem__(self, idx):
        sample = self.base[idx]
        # the base loaders mix torch tensors and numpy arrays
        for k in ('intri', 'joints2d', 'high_fps_joints2d', 'events', 'flows'):
            if k in sample and isinstance(sample[k], np.ndarray):
                sample[k] = torch.from_numpy(sample[k]).float()
        boxes = self._frame_boxes(sample, idx)
        crop_box = clip_union_bbox(boxes, pad_ratio=self.pad_ratio)

        if self.jitter > 0:
            size = crop_box[2]
            crop_box = crop_box + torch.tensor([
                float(np.random.uniform(-self.jitter, self.jitter)) * size,
                float(np.random.uniform(-self.jitter, self.jitter)) * size,
                float(np.random.uniform(-self.jitter, self.jitter)) * size,
            ])
            crop_box[2] = crop_box[2].clamp(0.15, 1.0)
            crop_box[0] = crop_box[0].clamp(crop_box[2] / 2, 1 - crop_box[2] / 2)
            crop_box[1] = crop_box[1].clamp(crop_box[2] / 2, 1 - crop_box[2] / 2)

        sample['events'] = crop_resize_volume(
            sample['events'], crop_box, self.out_size)
        if 'flows' in sample:
            sample['flows'] = crop_resize_volume(
                sample['flows'], crop_box, self.out_size, scale_values=True)

        sample['intri_full'] = sample['intri'].clone()
        sample['intri'] = adjust_intrinsics(
            sample['intri'], crop_box, self.img_size, self.out_size)

        sample['joints2d'] = remap_points_norm(sample['joints2d'], crop_box)
        if 'high_fps_joints2d' in sample:
            sample['high_fps_joints2d'] = remap_points_norm(
                sample['high_fps_joints2d'], crop_box)

        if 'raw_events' in sample:
            remapped = remap_raw_events(
                sample['raw_events'], crop_box, self.img_size, self.out_size)
            # event_breaks index into the per-clip event array; recompute
            # against the filtered stream by binning the surviving events
            # with the same per-frame boundaries.
            if 'event_breaks' in sample and remapped.shape[0] > 0:
                old = sample['raw_events']
                breaks = np.asarray(sample['event_breaks']).astype(np.int64)
                frame_id = np.searchsorted(breaks, np.arange(old.shape[0]),
                                           side='right')
                x0, y0 = crop_box[0] - crop_box[2] / 2, crop_box[1] - crop_box[2] / 2
                zoom = self.out_size / (crop_box[2] * self.img_size)
                xs = (old[:, 0] - x0 * self.img_size) * zoom
                ys = (old[:, 1] - y0 * self.img_size) * zoom
                keep = ((xs >= 0) & (xs < self.out_size) &
                        (ys >= 0) & (ys < self.out_size)).numpy()
                counts = np.bincount(frame_id[keep], minlength=len(breaks))
                sample['event_breaks'] = np.cumsum(counts).astype(
                    np.asarray(sample['event_breaks']).dtype)
            sample['raw_events'] = remapped
            sample['event_size'] = int(remapped.shape[0])

        sample['crop_box'] = crop_box
        return sample
