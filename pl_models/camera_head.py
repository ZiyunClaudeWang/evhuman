"""Weak-perspective camera head for crop-based translation recovery.

Follows the HMR 2.0 recipe: the network sees a subject-centred crop and
regresses a weak-perspective camera (s, tx, ty) for it. Combined with the
crop box and the full-frame intrinsics, this converts analytically to a
camera-frame root translation (see crop_utils.crop_cam_to_translation).

On fixed-distance lab captures the temporal GMP remains more accurate; this
head is the out-of-distribution fallback — it makes translation available
for captures at subject distances the GMP was never trained on, where a
learned depth prior would fail silently.
"""
import torch
import torch.nn as nn

from pl_data.crop_utils import crop_cam_to_translation


class CameraHead(nn.Module):
    """Per-frame (s, tx, ty) from the temporal encoder's features."""

    DEFAULT_SCALE = 0.9  # sigmoid-centred initial scale, subject fills crop

    def __init__(self, feat_dim=2048, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        # bias the scale toward a subject filling ~90% of the crop
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats):
        """
        Args:
            feats: [B, T, feat_dim] temporal features
        Returns:
            pred_cam: [B, T, 3] with s > 0
        """
        raw = self.net(feats)
        s = self.DEFAULT_SCALE * torch.exp(raw[..., 0:1])
        return torch.cat([s, raw[..., 1:3]], dim=-1)

    @staticmethod
    def to_translation(pred_cam, crop_box, intri_full, img_size=256):
        """Convert [B, T, 3] crop cameras to [B, T, 3] camera-frame
        translations.

        Args:
            crop_box: [B, 3] normalized (cx, cy, size), one crop per clip
            intri_full: [B, 4] full-frame intrinsics
        """
        B, T, _ = pred_cam.shape
        cam = pred_cam.reshape(B * T, 3)
        box = crop_box[:, None, :].expand(B, T, 3).reshape(B * T, 3)
        intr = intri_full[:, None, :].expand(B, T, 4).reshape(B * T, 4)
        trans = crop_cam_to_translation(cam, box, intr, img_size)
        return trans.view(B, T, 3)
