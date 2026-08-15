import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet18


class BBoxDetector(nn.Module):
    """Single-person bounding box detector for event camera data.

    Input: event volume [B, C, H, W] (e.g. 8-channel, 256x256)
    Output: normalized bounding box [B, 4] as (cx, cy, w, h) in [0, 1]
    """

    def __init__(self, input_channels=8, backbone='resnet18'):
        super().__init__()

        if backbone == 'resnet18':
            net = resnet18(weights=None)
            feat_dim = 512
        else:
            net = resnet34(weights=None)
            feat_dim = 512

        net.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # remove FC, keep avgpool

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] event volume
        Returns:
            bbox: [B, 4] as (cx, cy, w, h) normalized to [0, 1]
        """
        feat = self.backbone(x).squeeze(-1).squeeze(-1)
        bbox = self.head(feat)
        return bbox


def bbox_from_joints2d(joints2d, img_size=256, pad_ratio=0.2):
    """Compute bounding box from 2D joint positions.

    Args:
        joints2d: [B, T, J, 2] or [B, J, 2] in pixel coordinates [0, img_size]
        img_size: image dimension (assumes square)
        pad_ratio: padding around the tight bbox as fraction of bbox size

    Returns:
        bbox: [B, (T,) 4] as (cx, cy, w, h) normalized to [0, 1]
    """
    if joints2d.dim() == 4:
        B, T, J, _ = joints2d.shape
        flat = joints2d.view(B * T, J, 2)
        result = _compute_bbox(flat, img_size, pad_ratio)
        return result.view(B, T, 4)
    else:
        return _compute_bbox(joints2d, img_size, pad_ratio)


def _compute_bbox(joints2d, img_size, pad_ratio):
    """joints2d: [N, J, 2] in pixels → bbox: [N, 4] normalized."""
    x_min = joints2d[:, :, 0].min(dim=1).values
    x_max = joints2d[:, :, 0].max(dim=1).values
    y_min = joints2d[:, :, 1].min(dim=1).values
    y_max = joints2d[:, :, 1].max(dim=1).values

    w = x_max - x_min
    h = y_max - y_min
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    # square bbox: use max of w, h
    size = torch.max(w, h)
    pad = size * pad_ratio
    size = size + 2 * pad

    # normalize to [0, 1]
    cx = cx / img_size
    cy = cy / img_size
    size = size / img_size

    bbox = torch.stack([cx, cy, size, size], dim=1)
    return bbox.clamp(0, 1)


def bbox_iou(pred, target):
    """Compute IoU between two sets of (cx, cy, w, h) bboxes.

    Args:
        pred: [B, 4], target: [B, 4] — normalized (cx, cy, w, h)
    Returns:
        iou: [B]
    """
    # convert to (x1, y1, x2, y2)
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    tgt_x1 = target[:, 0] - target[:, 2] / 2
    tgt_y1 = target[:, 1] - target[:, 3] / 2
    tgt_x2 = target[:, 0] + target[:, 2] / 2
    tgt_y2 = target[:, 1] + target[:, 3] / 2

    inter_x1 = torch.max(pred_x1, tgt_x1)
    inter_y1 = torch.max(pred_y1, tgt_y1)
    inter_x2 = torch.min(pred_x2, tgt_x2)
    inter_y2 = torch.min(pred_y2, tgt_y2)

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    pred_area = pred[:, 2] * pred[:, 3]
    tgt_area = target[:, 2] * target[:, 3]
    union_area = pred_area + tgt_area - inter_area

    return inter_area / (union_area + 1e-6)
