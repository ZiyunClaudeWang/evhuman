"""Train a single-person bounding box detector on event camera data.

Uses the existing event volume images and derives GT bounding boxes from
the GT 2D joint annotations. The detector predicts (cx, cy, w, h) in
normalized [0,1] coordinates.

Usage:
    PYTHONPATH=NeMF/src:$PYTHONPATH python train_bbox_detector.py \
        --data_root data/mmhpsd \
        --event_folder data/mmhpsd_events/ \
        --epochs 20 --batch_size 32 --lr 1e-3
"""
import argparse
import collections
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from pl_models.bbox_detector import BBoxDetector, bbox_from_joints2d, bbox_iou


class BBoxDataset(torch.utils.data.Dataset):
    """Wraps the existing event HPE data loaders to yield (event_image, bbox) pairs.

    Each sample is a single event frame (not a clip), paired with the bbox
    derived from the GT 2D joints at the END of that frame's interval.
    """

    def __init__(self, data_dir, mode='train', img_size=256,
                 event_folder='data/mmhpsd_events/', our_data=False,
                 stride=1, skip=2):
        import pickle
        import joblib
        import cv2

        self.data_dir = data_dir
        self.img_size = img_size
        self.mode = mode

        if our_data:
            from event_hpe.our_data_loader import OursTrackingDataloader
            base_loader = OursTrackingDataloader(
                data_dir=data_dir, mode=mode, max_steps=8, num_steps=8, skip=skip,
                img_size=img_size, event_folder=event_folder,
                use_hmr_feats=False, use_flow=False)
        else:
            from event_hpe.data_loader import TrackingDataloader
            base_loader = TrackingDataloader(
                data_dir=data_dir, mode=mode, max_steps=8, num_steps=8, skip=skip,
                img_size=img_size, event_folder=event_folder,
                use_hmr_feats=False, use_flow=False)

        self.samples = []
        print(f'[BBoxDataset] Building index from {len(base_loader)} clips...')
        for idx in range(0, len(base_loader), stride):
            action, frame_idx = base_loader.all_clips[idx]
            self.samples.append(idx)

        self.base_loader = base_loader
        print(f'[BBoxDataset] {mode}: {len(self.samples)} samples (stride={stride})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # A few sequences have missing event PNGs; skip to a neighbour.
        for attempt in range(8):
            try:
                data = self.base_loader[self.samples[(idx + attempt) % len(self.samples)]]
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f'8 consecutive unreadable samples at index {idx}')
        events = data['events']         # [T, C, H, W]
        # The tracking loaders divide joints2d by img_size, so these are
        # normalized to [0, 1]; bbox_from_joints2d expects pixels.
        joints2d = data['joints2d'] * self.img_size   # [T, 24, 2] in pixels

        # Use middle frame's event image and corresponding joints
        mid = events.shape[0] // 2
        event_img = events[mid]          # [C, H, W]
        j2d = joints2d[mid]              # [24, 2]

        bbox = bbox_from_joints2d(j2d.unsqueeze(0), self.img_size).squeeze(0)  # [4]

        return event_img, bbox


def train(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    train_dataset = BBoxDataset(args.data_root, mode='train', img_size=args.img_size,
                                event_folder=args.event_folder, our_data=args.our_data,
                                stride=args.stride, skip=args.skip)
    val_dataset = BBoxDataset(args.data_root, mode='test', img_size=args.img_size,
                              event_folder=args.event_folder, our_data=args.our_data,
                              stride=max(1, args.stride // 2), skip=args.skip)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = BBoxDetector(input_channels=args.num_event_channels, backbone=args.backbone).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # per-iteration cosine decay (stepped every batch, not every epoch)
    total_iters = args.epochs * max(1, len(train_dataset) // args.batch_size)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)

    start_time_str = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    log_dir = os.path.join(args.result_dir, 'bbox_detector', start_time_str)
    writer = SummaryWriter(log_dir)
    print(f'[tensorboard] {log_dir}')

    best_iou = 0
    global_step = 0

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_losses = []
        train_ious = []

        for event_img, gt_bbox in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [train]'):
            event_img = event_img.to(device, dtype=torch.float32)
            gt_bbox = gt_bbox.to(device, dtype=torch.float32)

            pred_bbox = model(event_img)
            loss = F.smooth_l1_loss(pred_bbox, gt_bbox)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                iou = bbox_iou(pred_bbox, gt_bbox).mean()

            train_losses.append(loss.item())
            train_ious.append(iou.item())

            if global_step % 100 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/iou', iou.item(), global_step)
            global_step += 1

        avg_train_loss = np.mean(train_losses)
        avg_train_iou = np.mean(train_ious)

        # --- Validate ---
        model.eval()
        val_losses = []
        val_ious = []

        with torch.no_grad():
            for event_img, gt_bbox in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [val]'):
                event_img = event_img.to(device, dtype=torch.float32)
                gt_bbox = gt_bbox.to(device, dtype=torch.float32)

                pred_bbox = model(event_img)
                loss = F.smooth_l1_loss(pred_bbox, gt_bbox)
                iou = bbox_iou(pred_bbox, gt_bbox).mean()

                val_losses.append(loss.item())
                val_ious.append(iou.item())

        avg_val_loss = np.mean(val_losses)
        avg_val_iou = np.mean(val_ious)

        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/iou', avg_val_iou, epoch)

        print(f'Epoch {epoch+1}/{args.epochs}: '
              f'train_loss={avg_train_loss:.4f} train_iou={avg_train_iou:.3f} | '
              f'val_loss={avg_val_loss:.4f} val_iou={avg_val_iou:.3f}')

        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            save_path = os.path.join(log_dir, 'best_bbox_detector.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_iou': best_iou,
            }, save_path)
            print(f'  Saved best model (IoU={best_iou:.3f}) to {save_path}')

        # Save latest
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }, os.path.join(log_dir, 'latest_bbox_detector.pth'))

    writer.close()
    print(f'Training complete. Best val IoU: {best_iou:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train event-based human bounding box detector')

    parser.add_argument('--data_root', type=str, default='data/mmhpsd')
    parser.add_argument('--event_folder', type=str, default='data/mmhpsd_events/')
    parser.add_argument('--result_dir', type=str, default='outputs')
    parser.add_argument('--gpu_id', type=str, default='0')

    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_event_channels', type=int, default=8)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--our_data', action='store_true')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--stride', type=int, default=1,
                        help='index every Nth clip to shorten an epoch')
    parser.add_argument('--skip', type=int, default=2)

    args = parser.parse_args()
    train(args)
