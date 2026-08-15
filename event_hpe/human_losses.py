import torch
import torch.nn.functional as F

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def batch_rodrigues(theta):
    # theta N x 3
    # batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat2mat(quat)

def compute_mpjpe(pred, target):
    # [B, T, 24, 3]
    mpjpe = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))
    return mpjpe


# def compute_pa_mpjpe(pred, target):
#     B, T, _, _ = pred.size()
#     pred_hat = batch_compute_similarity_transform_torch(pred.view(-1, 24, 3), target.view(-1, 24, 3))
#     pa_mpjpe = torch.sqrt(torch.sum((pred_hat - target.view(-1, 24, 3)) ** 2, dim=-1))
#     return pa_mpjpe.view(B, T, 24)


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


def compute_losses(out, target):

    losses = {}

    # delta_tran_weight = loss_weights['delta_tran_weight']
    # delta_tran_loss = torch.mean(torch.pow(out['delta_tran'], 2))
    # losses['delta_trans'] = delta_tran_loss

    tran_loss = F.mse_loss(out['tran'], target['tran'])
    losses['tran'] = tran_loss

    theta_loss = rotation_loss(out['theta'].reshape(-1, 3), target['theta'].reshape(-1, 3))
    losses['theta'] = theta_loss

    joints3d_loss = F.mse_loss(out['joints3d'], target['joints3d'])
    losses['joints3d'] = joints3d_loss

    # pred_joints2d = torch.clamp(out['joints2d'], 0, 1)
    # joints2d_loss = F.mse_loss(pred_joints2d, target['joints2d'])
    # losses['joints2d'] = joints2d_loss

    return losses

def rotation_loss(pred_axis_angle, target_axis_angle, use_geodesic_loss=True):
    '''
    pred_axis_angle: [B, 3]
    target_axis_angle: [B, 3]
    '''
    target_rotmats = batch_rodrigues(target_axis_angle).view(-1, 3, 3)
    pred_rotmats = batch_rodrigues(pred_axis_angle).view(-1, 3, 3)

    if use_geodesic_loss:
        eps = 1e-6
        # square geodesic loss arccos[(Tr(R1R2^T) -1 )/2]
        trace_rrt = torch.sum(pred_rotmats * target_rotmats, dim=(-2, -1))  # [B, T, 24]
        # trace_rrt = torch.einsum('bij,bji->b', pred_rotmats, target_rotmats)
        degree_dif = torch.acos(torch.clamp(0.5 * (trace_rrt - 1), -1 + eps, 1 - eps))
        theta_loss = torch.mean(degree_dif)
    else:
        theta_loss = F.mse_loss(pred_rotmats, target_rotmats)

    return theta_loss


# def compute_flow_loss(verts, pred_flows, cam_intr, device):
#     # verts: [B, T+1, 6890, 3]
#     # pred_flows: [B, T, 2, H, W]
#     B, T, _, H, W = pred_flows.size()
#     verts_2d = projection_torch(verts, cam_intr)  # [B, T+1, 6890, 2]
#     verts_flow = verts_2d[:, 1:, :, :] - verts_2d[:, :-1, :, :]  # [B, T, 6890, 2] <t+1> - <t>

#     scale = torch.tensor([W, H], device=device)
#     flow_indices = 2. * verts_2d[:, :-1, :, :].detach().reshape(-1, 6890, 2) / scale - 1.  # [BT, 6890, 2]
#     flow_indices = F.pad(flow_indices, [0, 0, 0, H * W - 6890, 0, 0],
#                          mode='constant', value=-1).view(-1, H, W, 2)  # [BT, H, W, 2]
#     _flows = pred_flows.view(-1, 2, H, W)  # [BT, 2, H, W]
#     sampled_flow = F.grid_sample(_flows, flow_indices)  # [BT, 2, H, W]  (u, v)
#     sampled_flow = sampled_flow.permute(0, 2, 3, 1).view(B, T, H * W, 2)[:, :, :6890, :].clone()

#     # error = verts_flow - sampled_flow
#     # error = torch.clamp(error, min=-5, max=5)
#     # loss = torch.mean(torch.pow(error, 2))

#     # loss = torch.mean(1 - F.cosine_similarity(verts_flow, sampled_flow, dim=3))

#     valid = torch.norm(sampled_flow, p=2, dim=-1) > 4
#     sim = valid * (1 - F.cosine_similarity(verts_flow, sampled_flow, dim=3))  # [B, T, 6890]
#     loss = torch.mean(sim)
#     return loss




