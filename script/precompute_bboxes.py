"""Run the trained bbox detector over a tracking dataset and cache per-clip,
per-frame boxes for CroppedTrackingDataset.

Output: pkl mapping (action, frame_idx) -> [T, 4] float32 (cx, cy, w, h),
normalized to [0, 1].

Usage:
    PYTHONPATH=NeMF/src:$PYTHONPATH python script/precompute_bboxes.py \
        --data_root data/beahm --event_folder data/beahm_events/ \
        --detector outputs/bbox_detector/<ts>/best_bbox_detector.pth \
        --our_data --skip 8 --mode test --out bbox_cache_test.pkl
"""
import argparse
import os
import pickle
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pl_models.bbox_detector import BBoxDetector


class _EventsOnly(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        try:
            s = self.base[idx]
        except Exception:
            return idx, torch.zeros(0)
        return idx, s['events']  # [T, C, H, W]


def _collate(batch):
    return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--event_folder', type=str, required=True)
    p.add_argument('--detector', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--mode', type=str, default='test')
    p.add_argument('--our_data', action='store_true')
    p.add_argument('--skip', type=int, default=8)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--num_event_channels', type=int, default=8)
    p.add_argument('--backbone', type=str, default='resnet18')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--gpu_id', type=str, default='0')
    p.add_argument('--stride', type=int, default=1)
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}'
                          if torch.cuda.is_available() else 'cpu')

    if args.our_data:
        from event_hpe.our_data_loader import OursTrackingDataloader
        base = OursTrackingDataloader(
            data_dir=args.data_root, mode=args.mode, max_steps=8, num_steps=8,
            skip=args.skip, img_size=args.img_size,
            event_folder=args.event_folder, use_hmr_feats=False, use_flow=False)
    else:
        from event_hpe.data_loader import TrackingDataloader
        base = TrackingDataloader(
            data_dir=args.data_root, mode=args.mode, max_steps=8, num_steps=8,
            skip=args.skip, img_size=args.img_size,
            event_folder=args.event_folder, use_hmr_feats=False, use_flow=False)

    det = BBoxDetector(input_channels=args.num_event_channels,
                       backbone=args.backbone).to(device)
    ckpt = torch.load(args.detector, map_location=device)
    det.load_state_dict(ckpt['model_state_dict'])
    det.eval()
    print(f'[detector] {args.detector} (val_iou={ckpt.get("val_iou", "?")})')

    ds = _EventsOnly(base)
    subset = list(range(0, len(ds), args.stride))
    loader = DataLoader(torch.utils.data.Subset(ds, subset),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=_collate)

    cache = {}
    skipped = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            for idx, events in batch:
                if events.numel() == 0:
                    skipped += 1
                    continue
                T = events.shape[0]
                boxes = det(events.to(device, torch.float32))  # [T, 4]
                action, frame_idx = base.all_clips[idx]
                cache[(action, int(frame_idx))] = boxes.cpu().numpy()

    with open(args.out, 'wb') as f:
        pickle.dump(cache, f)
    print(f'[done] {len(cache)} clips cached to {args.out} '
          f'({skipped} unreadable clips skipped)')


if __name__ == '__main__':
    main()
