import pytorch_lightning as pl
import scipy
import nvdiffrast.torch as dr
import scipy.interpolate
from event_hpe.geometry import projection_torch
from lib.utils.visualizer import draw_kpts

import sys
sys.path.append('NeMF/src')
from NeMF.src.nemf.generative import Architecture
from NeMF.src.nemf.fk import ForwardKinematicsLayer
from NeMF.src.rotations import matrix_to_rotation_6d
from NeMF.src.utils import estimate_angular_velocity, estimate_linear_velocity

import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from vis_utils import gen_event_images
from event_hpe.geometry import rot6d_to_rotmat
from event_hpe.SMPL import batch_rodrigues

CAM_MOTION_DIM = 6


class EventHumanModel(pl.LightningModule):

    def __init__(self, args):
        super(EventHumanModel, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.global_z_dim = args.nemf.global_z
        self.local_z_dim = args.nemf.local_z
        self.clip_len = args.clip_len
        self.num_event_channels = args.num_event_channels
        self.batch_size = args.batch_size

        self.glctx = dr.RasterizeCudaContext()

        if args.our_data:
            self.smpl_dir = 'event_hpe/smpl_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
        else:
            self.smpl_dir = 'event_hpe/smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl'

        from event_hpe.SMPL import SMPL
        self.smpl = SMPL(self.smpl_dir, args.batch_size * args.clip_len)

        if args.hmr_model:
            from event_hpe.model import TemporalEncoder, resnet50_encoder
            use_flow = args.use_flow

            input_ch = self.num_event_channels + 2 * use_flow if use_flow else self.num_event_channels

            self.img_encoder = resnet50_encoder(input_ch)  # 2048

            # temporal encoder parameters
            n_layers = 1
            hidden_size = 2048
            bidirectional = False
            add_linear = False
            use_residual = True

            if args.abl_transformer:
                from event_hpe.models.transformer_causal import CausalTransformer
                self.tmp_encoder = CausalTransformer(
                    input_dim=hidden_size,
                    embed_dim=256,
                    n_heads=8,
                    mlp_dim=4 * 256,
                    depth=3,
                    use_cls=getattr(args, 'use_cls', False),
                    causal=not getattr(args, 'non_causal', False),
                )
            else:
                self.tmp_encoder = TemporalEncoder(
                    n_layers=n_layers,
                    hidden_size=hidden_size,
                    bidirectional=bidirectional,
                    add_linear=add_linear,
                    use_residual=use_residual,
                )

            from event_hpe.model import Regressor
            pose_dim = 24 * 6
            self.regressor = Regressor(pose_dim)

            self.feature_to_code = nn.Linear(2048, self.global_z_dim + self.local_z_dim)
            self.nemf = Architecture(args, 1)
            self.nemf.load(optimal=True)
            self.nemf.requires_grad = False
            self.nemf.eval()
            self.fk = ForwardKinematicsLayer(args, smpl_path=self.smpl_dir)

            from NeMF.src.arguments import Arguments
            from NeMF.src.nemf.global_motion import MyGlobalMotionPredictor
            gmp_args = Arguments(args.config_path, filename=args.pretrained_gmp)

            if args.pred_traj:
                self.gmp = MyGlobalMotionPredictor(gmp_args, 1)
        else:
            raise NotImplementedError

    def resize_smpl(self, size):
        from event_hpe.SMPL import SMPL
        self.smpl = SMPL(self.smpl_dir, size).to(self.device)

    def forward(self, batch, test=False, override_clip_length=-1, decode_ts=None):

        event_images = batch["events"]
        B, L, C, H, W = event_images.shape

        assert C == self.num_event_channels
        assert self.clip_len == L

        if test and batch['hidden_feats'].dim() == 3:
            batch['hidden_feats'] = batch['hidden_feats'][:, 0]
            batch['init_shape'] = batch['init_shape'][:, 0]

        init_shape = batch['init_shape']
        decode_length = self.clip_len

        if override_clip_length > 0:
            decode_length = override_clip_length

        events = event_images
        pred_flow = batch['flows']
        hidden_feats = batch['hidden_feats']

        if self.args.use_flow:
            x = torch.cat([events, pred_flow], dim=2).view(-1, events.size(2) + pred_flow.size(2), H, W)
        else:
            x = events.view(-1, events.size(2), H, W)

        x = self.img_encoder(x).view(B, L, 2048)
        x, hn = self.tmp_encoder(x, hidden_feats, return_hidden=True)  # [B, T, 2048]

        x = x.contiguous()
        hn = hn.contiguous().permute(1, 0, 2)

        delta_pose, delta_tran = self.regressor(x)  # pose [B, T, 24*6], tran [B, T, 3]

        # translation
        trans = init_shape[:, :, 0:3] + torch.cumsum(delta_tran, dim=1)
        trans = trans.unsqueeze(dim=2)  # [B, T, 1, 3]

        if decode_length != L:
            global_trans = F.interpolate(
                trans[:, :, 0, :].permute(0, 2, 1), (decode_length),
                align_corners=True, mode="linear"
            ).permute(0, 2, 1)[:, :, None, :]
        else:
            global_trans = trans

        # project hidden state to latent codes [B, z_g + z_l]
        code = self.feature_to_code(hn.reshape(B, -1))
        [z_g, z_l] = code.split([self.global_z_dim, self.local_z_dim], dim=1)

        if decode_ts is not None:
            output = self.nemf.decode(z_l, z_g=z_g, length=decode_length, step=1, decode_ts=decode_ts)
        else:
            output = self.nemf.decode(z_l, z_g=z_g, length=decode_length, step=1)
        pred_pose_mat = output['local_rotmat'].reshape(B, decode_length, 24, 3, 3)

        if self.args.ours_full_pose0:
            # convert pred_pose_mat into relative to the first pose
            gt_pose = batch['theta'].view(-1, 24, 3)
            gt_pose_rotmats = batch_rodrigues(gt_pose.view(-1, 3)).view(B, L, 24, 3, 3)

            root_orient_mat = rot6d_to_rotmat(output['root_orient']).view(B, decode_length, 1, 3, 3)  # [B, T, 1, 3, 3]
            pred_pose_mat = torch.cat([root_orient_mat, pred_pose_mat[:, :, 1:, ...]], dim=2)

            # left multiply
            pred_pose_rel_p1 = torch.matmul(pred_pose_mat[:, :1, :, :, :].permute(0, 1, 2, 4, 3), pred_pose_mat)

            init_root_rotmats = gt_pose_rotmats[:, 0, :, :, :]
            pred_pose_mat = torch.matmul(init_root_rotmats.unsqueeze(1), pred_pose_rel_p1)
            theta_rotmat = pred_pose_mat.contiguous()
        else:
            raise NotImplementedError

        beta = init_shape[:, :, 75:85].repeat(1, decode_length, 1)
        theta_rotmat = theta_rotmat.view(B * decode_length, 24, 3, 3)
        beta = beta.view(-1, 10)

        if self.args.pred_traj:
            assert self.args.ours_full_pose0
            b_size = z_l.shape[0]
            n_joints = 24

            # GMP inputs in camera coordinates (matching train_gmp.py)
            gmp_input_rotmat = theta_rotmat.view(B, decode_length, 24, 3, 3)
            pos_recon, _ = self.fk(gmp_input_rotmat.view(-1, n_joints, 3, 3))
            pos_recon = pos_recon.contiguous().view(b_size, -1, n_joints, 3)

            step = 1

            gmp_data = dict()
            gmp_data['rot6d'] = matrix_to_rotation_6d(gmp_input_rotmat)
            gmp_data['pos'] = pos_recon

            if self.args.test_high_fps:
                assert self.args.skip == 8
                assert pos_recon.shape[1] == 65

                dt = 1.0 / (self.args.data.fps * (decode_length - 1) / L) * step

                pos_down_sampled = pos_recon[:, ::8, :, :]
                gmp_input_rotmat_down_sampled = gmp_input_rotmat[:, ::8, :, :, :]

                angular_smooth = estimate_angular_velocity(gmp_input_rotmat_down_sampled, dt=dt * 8)
                velocity_smooth = estimate_linear_velocity(pos_down_sampled, dt=dt * 8)

                new_length = L * 8

                highfps_gmp_data = dict()
                highfps_gmp_data['pos'] = pos_down_sampled
                highfps_gmp_data['angular'] = angular_smooth
                highfps_gmp_data['velocity'] = velocity_smooth
                highfps_gmp_data['rot6d'] = gmp_data['rot6d'][:, ::8, :, :]

                gt_trans = batch['tran']
                highfps_gmp_data['origin'] = gt_trans[:, 0, 0, :]
                highfps_pred_data = self.gmp.predict(highfps_gmp_data, dt=dt * 8, no_height=True)

                low_fps_time = decode_ts[:, ::8].cpu()
                low_fps_trans = highfps_pred_data['trans'].cpu().numpy()[:, :, None, ...]

                inter = scipy.interpolate.interp1d(low_fps_time[0, :].numpy(), low_fps_trans, axis=1)
                trans = torch.tensor(
                    inter(decode_ts[0, :new_length].cpu().numpy()),
                    device=theta_rotmat.device
                )

                theta_rotmat = theta_rotmat.view(B, decode_length, 24, 3, 3)
                theta_rotmat = theta_rotmat[:, :new_length, ...].contiguous().view(B * new_length, 24, 3, 3)
                decode_length = new_length
                global_trans = trans

            else:
                dt = 1.0 / (self.args.data.fps * decode_length / L) * step
                gmp_data['angular'] = estimate_angular_velocity(gmp_input_rotmat.clone(), dt=dt)
                gmp_data['velocity'] = estimate_linear_velocity(pos_recon, dt=dt)

                gt_trans = batch['tran']
                origin = gt_trans[:, 0, 0, :]
                gmp_data['origin'] = origin

                pred_data = self.gmp.predict(gmp_data, dt=dt, no_height=True)
                trans = pred_data['trans'][:, :, None, ...]
                global_trans = trans

        verts, joints3d, _ = self.smpl(
            beta=beta[:theta_rotmat.shape[0]],
            theta=None,
            get_skin=True,
            rotmats=theta_rotmat
        )

        joints3d_canonical = joints3d.view(B, decode_length, 24, 3)
        joints3d = joints3d_canonical + global_trans.detach()

        gt_trans = batch['tran']

        if decode_length != L:
            gt_trans = F.interpolate(
                gt_trans[:, :, 0, :].permute(0, 2, 1), (decode_length),
                align_corners=True, mode="linear"
            ).permute(0, 2, 1)
            gt_trans = gt_trans[:, :, None, :]

        verts = verts.view(B, decode_length, -1, 3) + gt_trans.detach()

        pred_rotmat = theta_rotmat.view(B, decode_length, 24, 3, 3)
        init_shape = batch['init_shape']
        init_rotmats = batch_rodrigues(init_shape[:, 0, 3:75].reshape(-1, 3)).view(B, 24, 3, 3)
        init_verts, _, _ = self.smpl(
            beta=init_shape[:, 0, 75:85],
            theta=None,
            get_skin=True,
            rotmats=init_rotmats
        )

        init_verts = (init_verts + init_shape[:, :, 0:3]).unsqueeze(1)
        verts = torch.cat([init_verts, verts], dim=1)

        cam_intr = batch['intri']
        joints2d = projection_torch(joints3d, cam_intr, H=256, W=256)

        # GT meshes are only defined at the labelled keyframes, so skip this
        # when the motion field is queried at a different temporal resolution.
        if self.args.contrast_loss > 0 and decode_length == L:
            with torch.no_grad():
                gt_pose = batch['theta'].view(-1, 24, 3)
                gt_pose_rotmats = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)
                gt_verts, _, _ = self.smpl(
                    beta=beta,
                    get_skin=True,
                    rotmats=gt_pose_rotmats
                )

                gt_verts = gt_verts.view(B, L, -1, 3) + gt_trans
                gt_verts = torch.cat([init_verts, gt_verts], dim=1)

        out_dict = {
            "joints3d": joints3d,
            "joints2d": joints2d,
            "verts": verts,
            "trans": global_trans,
            "pred_rotmats": pred_rotmat,
            "beta": beta,
            'cam_intr': cam_intr,
        }
        if self.args.contrast_loss > 0:
            out_dict['gt_verts'] = gt_verts

        return out_dict

    @torch.no_grad()
    def plot_human(self, batch, output, prefix="train", step=0):
        B, T, _, H, W = batch['events'].shape
        event_images = batch['events'].view(-1, self.num_event_channels, H, W)
        base_images = gen_event_images(event_images, prefix='Human', device="cuda", clamp_val=2., normalize_events=True)

        plot_images = []
        pred_joints_2d = output['joints2d'].view(-1, 24, 2) * 256
        gt_keypoints = batch['joints2d'].view(-1, 24, 2) * 256

        gt_trans = batch['tran'].view(-1, 3)

        init_betas = batch['init_shape'][:, :, -10:].repeat(1, T, 1).view(-1, 10)

        # project 3d keypoints
        gt_rotmat = batch_rodrigues(batch['theta'].view(B * T * 24, 3)).view(B * T, 24, 3, 3)
        _, joints3d, _ = self.smpl(
            beta=init_betas,
            get_skin=True,
            rotmats=gt_rotmat
        )
        joints3d = joints3d.view(-1, 24, 3) + gt_trans[:, None, :]

        cam_intr = batch['intri']
        gt_joints3d_keypoints = projection_torch(joints3d.view(B, T, 24, 3), cam_intr, H=H, W=W) * 256
        gt_joints3d_keypoints = gt_joints3d_keypoints.view(-1, 24, 2)

        for i, image in enumerate(base_images):
            image = image.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            ev_image = draw_kpts(image, pred_joints_2d[i, ...].cpu(), color=(0, 0, 255), r=1, thickness=2, draw_id=False)
            ev_image = draw_kpts(ev_image, gt_joints3d_keypoints[i, ...].cpu(), color=(0, 255, 255), r=2, thickness=2, draw_id=False)
            ev_image = draw_kpts(ev_image, gt_keypoints[i, ...].cpu(), color=(255, 0, 0), r=1, thickness=2, draw_id=False)
            ev_image = torch.tensor(ev_image).permute(2, 0, 1)
            plot_images.append(ev_image)

        pred_images = torch.stack(plot_images, dim=0)
        return pred_images

    def compute_flow(self, v0, v1, faces, cam_intrinsics, H, W, return_mask=False):
        '''
        Args:
            vertices: [B, N, 3]
            faces: [F, 3]
            cam_intrinsics: [B, 4]
            H: int
            W: int
            return_mask: if True, also return the [B, H, W] silhouette mask
        '''
        flow = projection_torch(v1[:, None, ...], cam_intrinsics, H, W) \
                                - projection_torch(v0[:, None, ...], cam_intrinsics, H, W)

        faces = faces.to(torch.int32)
        vv = self.batch_convert_to_clip_space(v0, cam_intrinsics, 0.1, 20)
        ff = faces.contiguous()
        rast, _ = dr.rasterize(self.glctx, vv, ff, resolution=(H, W))
        out, _ = dr.interpolate(flow[:, 0, ...], rast, ff)

        if return_mask:
            mask = (rast[..., 3] > 0).float()  # [B, H, W]
            return out, mask
        return out

    @staticmethod
    def batch_convert_to_clip_space(batch_vertices, batch_intrinsics, z_near, z_far, H=256, W=256):
        """
        Converts a batch of 3D vertices to clip space using a batch of camera intrinsics.

        Args:
            batch_vertices (torch.Tensor): Batch of input vertices in 3D space, shape (B, N, 3).
            batch_intrinsics (torch.Tensor): Batch of intrinsics, shape (B, 4), where each row is [fx, fy, cx, cy].
            z_near (float): Near clipping plane distance.
            z_far (float): Far clipping plane distance.

        Returns:
            torch.Tensor: Batch of vertices in clip space, shape (B, N, 4).
        """
        fx = batch_intrinsics[:, 0, None]
        fy = batch_intrinsics[:, 1, None]
        cx = batch_intrinsics[:, 2, None]
        cy = batch_intrinsics[:, 3, None]

        u = batch_vertices[:, :, 0] / batch_vertices[:, :, 2] * fx + cx
        v = batch_vertices[:, :, 1] / batch_vertices[:, :, 2] * fy + cy

        z = batch_vertices[:, :, 2]

        x_ndc = 2 * u / W - 1
        y_ndc = (1 - 2 * v / H) * -1
        z_ndc = (z_far + z_near) / (z_far - z_near) + 2 * z_far * z_near / (z_near - z_far * z)
        clip_space_vertices = torch.stack([x_ndc, y_ndc, z_ndc, torch.ones_like(z_ndc)], dim=2)
        return clip_space_vertices

    @staticmethod
    def compute_IWE_v2(events, flow, mask=None, t_ref=1):
        """Improved IWE with polarity separation and silhouette masking.

        Args:
            events: [N, 4] (x, y, t, p)
            flow: [H, W, 2]
            mask: [H, W] silhouette mask (optional)
            t_ref: reference time for warping
        Returns:
            loss_val: scalar contrast loss (negative variance, higher = sharper)
        """
        img_size = flow.shape[0]
        t_norm = (events[:, 2] - events[0, 2]) / (events[-1, 2] - events[0, 2] + 1e-6)

        x_norm = (events[:, 0] - img_size / 2) / (img_size / 2)
        y_norm = (events[:, 1] - img_size / 2) / (img_size / 2)

        sampled_flow = F.grid_sample(
            flow.permute(2, 0, 1)[None, ...],
            torch.stack([x_norm, y_norm], dim=1)[None, None, ...],
            mode='bilinear', align_corners=True
        )

        flow_x = sampled_flow[0, 0, 0, :] * img_size
        flow_y = sampled_flow[0, 1, 0, :] * img_size

        x_warped = events[:, 0] + flow_x * (t_ref - t_norm)
        y_warped = events[:, 1] + flow_y * (t_ref - t_norm)

        polarity = events[:, 3]

        # Split into positive and negative polarity IWEs
        pos_mask = polarity > 0
        neg_mask = polarity <= 0

        total_var = 0.0
        count = 0

        for pmask in [pos_mask, neg_mask]:
            if pmask.sum() < 10:
                continue
            ev_input = torch.stack([y_warped[pmask], x_warped[pmask],
                                    t_norm[pmask], polarity[pmask]], dim=1)
            iwe = bilinear_vote_tensor(ev_input, weight=1.0, img_size=img_size)
            iwe = iwe.squeeze(0)  # [H, W]

            if mask is not None:
                iwe = iwe * mask
                n_pixels = mask.sum().clamp(min=1)
            else:
                n_pixels = img_size * img_size

            mean = iwe.sum() / n_pixels
            var = ((iwe - mean) ** 2 * (mask if mask is not None else 1.0)).sum() / n_pixels
            total_var += var
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=flow.device, requires_grad=True)

        return total_var / count

    @staticmethod
    def compute_contrast_loss_v3(events, flow, mask, t_ref=1):
        """Contrast loss using gradient-based sharpness in the human region.

        Instead of global IWE variance, computes the squared gradient magnitude
        (Laplacian sharpness) of the IWE within the mesh bounding box. This is
        more sensitive to edge sharpness than global variance.

        Args:
            events: [N, 4] (x, y, t, p)
            flow: [H, W, 2] in normalized coordinates
            mask: [H, W] binary silhouette mask
            t_ref: reference time for warping
        Returns:
            sharpness: scalar (higher = sharper IWE = better flow)
        """
        img_size = flow.shape[0]
        if events.shape[0] < 20:
            return torch.tensor(0.0, device=flow.device, requires_grad=True)

        t_norm = (events[:, 2] - events[0, 2]) / (events[-1, 2] - events[0, 2] + 1e-6)

        x_norm = (events[:, 0] - img_size / 2) / (img_size / 2)
        y_norm = (events[:, 1] - img_size / 2) / (img_size / 2)

        sampled_flow = F.grid_sample(
            flow.permute(2, 0, 1)[None, ...],
            torch.stack([x_norm, y_norm], dim=1)[None, None, ...],
            mode='bilinear', align_corners=True
        )

        flow_x = sampled_flow[0, 0, 0, :] * img_size
        flow_y = sampled_flow[0, 1, 0, :] * img_size

        x_warped = events[:, 0] + flow_x * (t_ref - t_norm)
        y_warped = events[:, 1] + flow_y * (t_ref - t_norm)

        # Build IWE
        ev_input = torch.stack([y_warped, x_warped, t_norm, events[:, 3]], dim=1)
        iwe = bilinear_vote_tensor(ev_input, weight=1.0, img_size=img_size).squeeze(0)

        # Compute bounding box of the mask with padding
        mask_ys = torch.where(mask.sum(dim=1) > 0)[0]
        mask_xs = torch.where(mask.sum(dim=0) > 0)[0]
        if len(mask_ys) == 0 or len(mask_xs) == 0:
            return torch.tensor(0.0, device=flow.device, requires_grad=True)

        pad = 15
        y1 = max(0, mask_ys[0].item() - pad)
        y2 = min(img_size, mask_ys[-1].item() + pad)
        x1 = max(0, mask_xs[0].item() - pad)
        x2 = min(img_size, mask_xs[-1].item() + pad)

        # Crop IWE to human region
        iwe_crop = iwe[y1:y2, x1:x2]

        if iwe_crop.numel() < 4:
            return torch.tensor(0.0, device=flow.device, requires_grad=True)

        # Squared gradient magnitude (Sobel-like sharpness)
        # Using finite differences for differentiability
        dx = iwe_crop[:, 1:] - iwe_crop[:, :-1]
        dy = iwe_crop[1:, :] - iwe_crop[:-1, :]
        grad_mag = dx[:min(dx.shape[0], dy.shape[0]), :] ** 2 + \
                   dy[:, :min(dx.shape[1], dy.shape[1])] ** 2

        # Also compute variance in the cropped region for combined metric
        n_pix = iwe_crop.numel()
        mean_val = iwe_crop.sum() / n_pix
        variance = ((iwe_crop - mean_val) ** 2).sum() / n_pix

        # Combined: gradient sharpness + variance
        sharpness = grad_mag.mean() + variance

        return sharpness

    @staticmethod
    def compute_IWE(events, flow, t_ref=1):
        '''
        Args:
            events: [N, 4]
            flow: [H, W, 2]
        '''
        img_size = flow.shape[0]
        assert img_size == 256

        t_norm = (events[:, 2] - events[0, 2]) / (events[-1, 2] - events[0, 2])

        x_norm = (events[:, 0] - img_size / 2) / (img_size / 2)
        y_norm = (events[:, 1] - img_size / 2) / (img_size / 2)

        flow = F.grid_sample(
            flow.permute(2, 0, 1)[None, ...],
            torch.stack([x_norm, y_norm], dim=1)[None, None, ...],
            mode='bilinear', align_corners=True
        )

        flow_x = flow[0, 0, 0, :] * img_size
        flow_y = flow[0, 1, 0, :] * img_size

        x_warped = events[:, 0] + flow_x * (t_ref - t_norm)
        y_warped = events[:, 1] + flow_y * (t_ref - t_norm)

        # compute IWE from warped events
        event_input = torch.stack([y_warped, x_warped, t_norm, events[:, 3]], dim=1)

        iwe = bilinear_vote_tensor(event_input, weight=1.0, img_size=img_size)
        return iwe


def bilinear_vote_tensor(events: torch.Tensor, weight: float = 1.0, img_size: int = 256, outer_padding: int = 0):
    """Tensor version of `bilinear_vote_numpy().`

    Args:
        events (torch.Tensor) ... [(b,) n_events, 4] Batch of events. 4 is (x, y, t, p). Attention that (x, y) could float.
        weight (float or torch.Tensor) ... Weight to multiply to the voting value.
            If scalar, the weight is all the same among events.
            If it's array-like, it should be the shape of [(b,) n_events].
            Defaults to 1.0.

    Returns:
        image ... [(b,) H, W]. Each index indicates the bilinear vote result. If the outer_padding is set,
            the return size will be [H + outer_padding, W + outer_padding].
    """
    if type(weight) == torch.Tensor:
        assert weight.shape == events.shape[:-1]
    if len(events.shape) == 2:
        events = events[None, ...]  # 1 x n x 4

    ph, pw = outer_padding, outer_padding
    h, w = img_size, img_size
    nb = len(events)
    image = events.new_zeros((nb, h * w))

    floor_xy = torch.floor(events[..., :2] + 1e-6)
    floor_to_xy = events[..., :2] - floor_xy
    floor_xy = floor_xy.long()

    x1 = floor_xy[..., 1] + pw
    y1 = floor_xy[..., 0] + ph
    inds = torch.cat(
        [
            x1 + y1 * w,
            x1 + (y1 + 1) * w,
            (x1 + 1) + y1 * w,
            (x1 + 1) + (y1 + 1) * w,
        ],
        dim=-1,
    )  # [(b, ) n_events x 4]
    inds_mask = torch.cat(
        [
            (0 <= x1) * (x1 < w) * (0 <= y1) * (y1 < h),
            (0 <= x1) * (x1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
            (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1) * (y1 < h),
            (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
        ],
        axis=-1,
    )

    w_pos0 = (1 - floor_to_xy[..., 0]) * (1 - floor_to_xy[..., 1]) * weight
    w_pos1 = floor_to_xy[..., 0] * (1 - floor_to_xy[..., 1]) * weight
    w_pos2 = (1 - floor_to_xy[..., 0]) * floor_to_xy[..., 1] * weight
    w_pos3 = floor_to_xy[..., 0] * floor_to_xy[..., 1] * weight
    vals = torch.cat([w_pos0, w_pos1, w_pos2, w_pos3], dim=-1)  # [(b,) n_events x 4]

    inds = (inds * inds_mask).long()
    vals = vals * inds_mask
    image.scatter_add_(1, inds, vals)
    return image.reshape((nb,) + (img_size, img_size)).squeeze()
