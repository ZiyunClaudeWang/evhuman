"""Crop geometry for detection-based, HMR-style training on event volumes.

Conventions (matching the tracking loaders):
- boxes are (cx, cy, size, size), normalized to [0, 1] of the full frame
- joints2d in samples are normalized to [0, 1] (loader divides by img_size)
- intrinsics are (fx, fy, px, py) in full-frame pixels
- projection_torch(xyz, intri, H, W) returns coordinates normalized by H, W

A crop is described by (x0, y0, s) in full-frame pixels: the square
[x0, x0+s) x [y0, y0+s) resized to out_size x out_size.
"""
import numpy as np
import torch
import torch.nn.functional as F


def clip_union_bbox(boxes, pad_ratio=0.1, min_size=0.2):
    """Union of per-frame boxes -> one square crop box for the whole clip.

    A single stable crop per clip keeps the temporal encoder's framing
    consistent, so image motion still reflects subject motion.

    Args:
        boxes: [T, 4] tensor, (cx, cy, w, h) normalized
        pad_ratio: extra padding as a fraction of the union size
        min_size: lower bound on crop size (normalized)
    Returns:
        (cx, cy, size) tensor, normalized, size clamped to keep box in frame
    """
    x0 = (boxes[:, 0] - boxes[:, 2] / 2).min()
    y0 = (boxes[:, 1] - boxes[:, 3] / 2).min()
    x1 = (boxes[:, 0] + boxes[:, 2] / 2).max()
    y1 = (boxes[:, 1] + boxes[:, 3] / 2).max()

    size = torch.max(x1 - x0, y1 - y0) * (1 + 2 * pad_ratio)
    size = size.clamp(min=min_size, max=1.0)
    cx = ((x0 + x1) / 2).clamp(size / 2, 1 - size / 2)
    cy = ((y0 + y1) / 2).clamp(size / 2, 1 - size / 2)
    return torch.stack([cx, cy, size])


def crop_box_to_pixels(crop_box, img_size):
    """(cx, cy, size) normalized -> (x0, y0, s) in pixels."""
    cx, cy, size = crop_box[0], crop_box[1], crop_box[2]
    s = size * img_size
    x0 = cx * img_size - s / 2
    y0 = cy * img_size - s / 2
    return x0, y0, s


def crop_resize_volume(vol, crop_box, out_size, scale_values=False):
    """Crop a [T, C, H, W] volume with a normalized (cx, cy, size) box and
    resize to out_size. grid_sample handles fractional-pixel boxes.

    Args:
        scale_values: multiply values by the zoom factor. Use for flow images,
            whose values are displacements in the resized pixel space.
    """
    T, C, H, W = vol.shape
    x0, y0, s = crop_box_to_pixels(crop_box, H)

    ys = torch.linspace(0, 1, out_size, device=vol.device)
    xs = torch.linspace(0, 1, out_size, device=vol.device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    # sample locations in full-frame pixels -> [-1, 1] grid coords
    sx = (x0 + gx * s) / (W - 1) * 2 - 1
    sy = (y0 + gy * s) / (H - 1) * 2 - 1
    grid = torch.stack([sx, sy], dim=-1)[None].expand(T, -1, -1, -1)

    out = F.grid_sample(vol, grid, mode='bilinear',
                        padding_mode='zeros', align_corners=True)
    if scale_values:
        out = out * (out_size / s)
    return out


def adjust_intrinsics(intri, crop_box, img_size, out_size):
    """Full-frame (fx, fy, px, py) -> intrinsics of the crop camera."""
    x0, y0, s = crop_box_to_pixels(crop_box, img_size)
    zoom = out_size / s
    fx, fy, px, py = intri[0], intri[1], intri[2], intri[3]
    return torch.stack([fx * zoom, fy * zoom, (px - x0) * zoom, (py - y0) * zoom])


def remap_points_norm(pts, crop_box):
    """Remap [..., 2] points normalized in the full frame into normalized
    crop coordinates."""
    cx, cy, size = crop_box[0], crop_box[1], crop_box[2]
    x0 = cx - size / 2
    y0 = cy - size / 2
    out = pts.clone()
    out[..., 0] = (pts[..., 0] - x0) / size
    out[..., 1] = (pts[..., 1] - y0) / size
    return out


def remap_raw_events(events, crop_box, img_size, out_size):
    """Transform raw events (x, y, t, p) in full-frame pixels into crop
    pixels, dropping events outside the crop."""
    x0, y0, s = crop_box_to_pixels(crop_box, img_size)
    zoom = out_size / s
    x = (events[:, 0] - x0) * zoom
    y = (events[:, 1] - y0) * zoom
    keep = (x >= 0) & (x < out_size) & (y >= 0) & (y < out_size)
    out = events[keep].clone()
    out[:, 0] = x[keep]
    out[:, 1] = y[keep]
    return out


def crop_cam_to_translation(pred_cam, crop_box, intri, img_size):
    """HMR 2.0-style conversion: weak-perspective camera predicted in the
    crop -> translation in the full camera frame.

    The head predicts pred_cam = (s, tx, ty) such that, in the crop,
    2D ~ s * (X + [tx, ty]) for the orthographic projection of the
    zero-centred body. Depth follows from matching the crop's pixel scale
    to the full-frame focal length:

        z = 2 * f / (s * b)        (b = crop size in full-frame pixels)

    Args:
        pred_cam: [B, 3] (s, tx, ty), s > 0
        crop_box: [B, 3] normalized (cx, cy, size)
        intri: [B, 4] full-frame (fx, fy, px, py)
        img_size: full-frame resolution
    Returns:
        trans: [B, 3] camera-frame translation of the body root
    """
    s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
    fx, fy, px, py = intri[:, 0], intri[:, 1], intri[:, 2], intri[:, 3]
    b = crop_box[:, 2] * img_size                      # crop size, pixels
    cx = crop_box[:, 0] * img_size                     # crop centre, pixels
    cy = crop_box[:, 1] * img_size

    z = 2 * fx / (s * b + 1e-9)
    x = tx + (cx - px) * z / fx
    y = ty + (cy - py) * z / fy
    return torch.stack([x, y, z], dim=1)
