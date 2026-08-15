import torch
import torch.nn.functional as F
from event_hpe.human_losses import batch_rodrigues
from event_hpe.geometry import projection_torch, batch_compute_similarity_transform_torch


def compute_mpjpe(pred, target):
    # [B, T, 24, 3]
    mpjpe = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))
    return mpjpe


def compute_pa_mpjpe(pred, target):
    B, T, _, _ = pred.size()
    pred_hat = batch_compute_similarity_transform_torch(pred.view(-1, 24, 3), target.view(-1, 24, 3))
    pa_mpjpe = torch.sqrt(torch.sum((pred_hat - target.view(-1, 24, 3)) ** 2, dim=-1))
    return pa_mpjpe.view(B, T, 24)


def compute_pelvis_mpjpe(pred, target):
    # [B, T, 24, 3]
    left_heap_idx = 1
    right_heap_idx = 2
    pred_pel = (pred[:, :, left_heap_idx:left_heap_idx+1, :] + pred[:, :, right_heap_idx:right_heap_idx+1, :]) / 2
    pred = pred - pred_pel
    target_pel = (target[:, :, left_heap_idx:left_heap_idx+1, :] + target[:, :, right_heap_idx:right_heap_idx+1, :]) / 2
    target = target - target_pel
    pel_mpjpe = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))
    return pel_mpjpe


def compute_pck(pred, target):
    pel_mpjpe = compute_pelvis_mpjpe(pred, target)
    pck = pel_mpjpe < 0.1
    return pck


def compute_pck_head(pred, target):
    # 0.5 head PCKh@0.5
    pel_mpjpe = compute_pelvis_mpjpe(pred, target)  # [B, T, 24]
    neck_idx = 12
    head_idx = 15
    thre = 0.5 * 2 * torch.sqrt(torch.sum(
        (target[:, :, neck_idx:neck_idx+1, :] - target[:, :, head_idx:head_idx+1, :]) ** 2, dim=-1))
    pck = pel_mpjpe < thre
    return pck


def compute_pck_torso(pred, target):
    # 0.2 torso PCK@0.2
    pel_mpjpe = compute_pelvis_mpjpe(pred, target)
    neck_idx = 12
    pel_idx = 0
    thre = 0.2 * torch.sqrt(torch.sum(
        (target[:, :, neck_idx:neck_idx + 1, :] - target[:, :, pel_idx:pel_idx + 1, :]) ** 2, dim=-1))
    pck = pel_mpjpe < thre
    return pck


def compute_losses(out, target, mse_func, device, args):
    losses = {}
    if args.delta_tran_loss > 0:
        delta_tran_loss = torch.mean(torch.pow(out['delta_tran'], 2))
        losses['delta_tran'] = args.delta_tran_loss * delta_tran_loss
    else:
        losses['delta_tran'] = torch.tensor([0], device=device).float()

    if args.tran_loss > 0:
        tran_loss = mse_func(out['trans'], target['tran'])
        losses['tran'] = args.tran_loss * tran_loss
    else:
        losses['tran'] = torch.tensor([0], device=device).float()

    if args.theta_loss > 0:
        B, T, _ = target['theta'].size()
        target_rotmats = batch_rodrigues(target['theta'].view(-1, 3)).view(B, T, 24, 3, 3)

        if args.use_geodesic_loss:
            eps = 1e-6
            # square geodesic loss arccos[(Tr(R1R2^T) -1 )/2]
            trace_rrt = torch.sum(out['pred_rotmats'] * target_rotmats, dim=(-2, -1))  # [B, T, 24]
            degree_dif = torch.acos(torch.clamp(0.5 * (trace_rrt - 1), -1 + eps, 1 - eps))
            theta_loss = torch.mean(degree_dif)

            for i in range(24):
                losses['theta_' + str(i)] = degree_dif[:, :, i].mean()
        else:
            theta_loss = mse_func(out['pred_rotmats'], target_rotmats)
        losses['theta'] = args.theta_loss * theta_loss
        for i in range(24):
            if 'theta_' + str(i) not in losses:
                losses['theta_' + str(i)] = torch.tensor(0.0, device=device)
    else:
        losses['theta'] = torch.tensor([0], device=device).float()
        for i in range(24):
            losses['theta_' + str(i)] = torch.tensor(0.0, device=device)

    if args.joints3d_loss > 0:
        joints3d_loss = mse_func(out['joints3d'], target['joints3d'])
        losses['joints3d'] = args.joints3d_loss * joints3d_loss
    else:
        losses['joints3d'] = torch.tensor([0], device=device).float()

    if args.joints2d_loss > 0:
        pred_joints2d = torch.clamp(out['joints2d'], 0, 1)
        joints2d_loss = mse_func(pred_joints2d, target['joints2d'])
        losses['joints2d'] = args.joints2d_loss * joints2d_loss
    else:
        losses['joints2d'] = torch.tensor([0], device=device).float()

    if args.flow_loss > 0:
        flow_loss = compute_flow_loss(out['verts'], target['flows'], out['cam_intr'], device)
        losses['flow'] = args.flow_loss * flow_loss
    else:
        losses['flow'] = torch.tensor([0], device=device).float()

    return losses


def compute_flow_loss(verts, pred_flows, cam_intr, device):
    # verts: [B, T+1, 6890, 3]
    # pred_flows: [B, T, 2, H, W]
    B, T, _, H, W = pred_flows.size()
    verts_2d = projection_torch(verts, cam_intr)  # [B, T+1, 6890, 2]
    verts_flow = verts_2d[:, 1:, :, :] - verts_2d[:, :-1, :, :]  # [B, T, 6890, 2] <t+1> - <t>

    scale = torch.tensor([W, H], device=device)
    flow_indices = 2. * verts_2d[:, :-1, :, :].detach().reshape(-1, 6890, 2) / scale - 1.  # [BT, 6890, 2]
    flow_indices = F.pad(flow_indices, [0, 0, 0, H * W - 6890, 0, 0],
                         mode='constant', value=-1).view(-1, H, W, 2)  # [BT, H, W, 2]
    _flows = pred_flows.view(-1, 2, H, W)  # [BT, 2, H, W]
    sampled_flow = F.grid_sample(_flows, flow_indices)  # [BT, 2, H, W]  (u, v)
    sampled_flow = sampled_flow.permute(0, 2, 3, 1).view(B, T, H * W, 2)[:, :, :6890, :].clone()

    valid = torch.norm(sampled_flow, p=2, dim=-1) > 4
    sim = valid * (1 - F.cosine_similarity(verts_flow, sampled_flow, dim=3))  # [B, T, 6890]
    loss = torch.mean(sim)

    if torch.isnan(loss):
        pass  # NaN flow loss
        loss = torch.tensor(0., device=verts.device, requires_grad=True)
    return loss

