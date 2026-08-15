import argparse
import collections
import os
import time

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.utils import flow_to_image
from tqdm import tqdm

from event_hpe.loss_funcs import (
    compute_losses, compute_mpjpe, compute_pa_mpjpe,
    compute_pelvis_mpjpe, compute_pck,
)
from pl_data.event_hpe_dm import EventHPEDataModule
from pl_models.event_human_model import EventHumanModel


def custom_collate_fn(batch):
    """Collate that concatenates variable-length raw_events along dim=0."""
    batch_dict = {key: [d[key] for d in batch] for key in batch[0]}
    if 'raw_events' in batch_dict:
        batch_dict['raw_events'] = torch.cat(batch_dict['raw_events'], dim=0)
    for key in batch_dict:
        if key != 'raw_events':
            batch_dict[key] = default_collate(batch_dict[key])
    return batch_dict


def compute_contrast(model, data, out, flow_verts, args, do_visualization):
    """Compute contrast maximization loss via image of warped events (IWE).

    Warps events using the predicted optical flow between two consecutive
    vertex frames and maximises the variance of the resulting IWE.

    Returns a dict with 'loss', 'all_iwes', and (when do_visualization is
    True) 'no_warp_iwes', 'gt_iwes', 'flow', 'gt_flow' for tensorboard.
    """
    batch_breaks = np.cumsum(data['event_size'].cpu().numpy()).astype(int)
    ref_frame = np.random.randint(0, flow_verts.shape[1] - 1)
    faces_tensor = torch.tensor(model.smpl.faces.astype(int)).to(flow_verts.device)

    flow = model.compute_flow(
        flow_verts[:, ref_frame, ...], flow_verts[:, ref_frame + 1, ...],
        faces_tensor, data['intri'], args.img_size, args.img_size,
    )

    gt_flow = None
    if do_visualization:
        gt_verts = out['gt_verts']
        gt_flow = model.compute_flow(
            gt_verts[:, ref_frame, ...], gt_verts[:, ref_frame + 1, ...],
            faces_tensor, data['intri'], args.img_size, args.img_size,
        )

    all_iwes = []
    no_warp_iwes = []
    gt_iwes = []

    for bb in range(batch_breaks.shape[0]):
        bstart = 0 if bb == 0 else batch_breaks[bb - 1]
        raw_events_one_batch = data['raw_events'][bstart:batch_breaks[bb]]
        time_breaks = data['event_breaks'].int()
        event_start = 0 if ref_frame == 0 else time_breaks[bb, ref_frame - 1]
        events_to_warp = raw_events_one_batch[event_start:time_breaks[bb, ref_frame]]

        all_iwes.append(model.compute_IWE(events_to_warp, flow[bb]))

        if do_visualization:
            no_warp_iwes.append(model.compute_IWE(events_to_warp, torch.zeros_like(flow[bb])))
            gt_iwes.append(model.compute_IWE(events_to_warp, gt_flow[bb]))

    all_iwes = torch.stack(all_iwes, dim=0)
    contrast_loss = args.contrast_loss * (-1) * torch.var(all_iwes, dim=(1, 2)).mean()

    return {
        'loss': contrast_loss,
        'all_iwes': all_iwes,
        'no_warp_iwes': no_warp_iwes,
        'gt_iwes': gt_iwes,
        'flow': flow,
        'gt_flow': gt_flow,
    }


def compute_contrast_v2(model, data, out, flow_verts, args, do_visualization):
    """Improved contrast loss: cropped human region, gradient sharpness, wider gaps.

    Key improvements:
    1. Uses wider temporal gaps (not just consecutive frames) for larger flow
    2. Crops IWE to mesh bounding box for focused signal
    3. Gradient-based sharpness (Sobel) + variance for edge sensitivity
    4. Multiple frame pairs per iteration
    """
    batch_breaks = np.cumsum(data['event_size'].cpu().numpy()).astype(int)
    faces_tensor = torch.tensor(model.smpl.faces.astype(int)).to(flow_verts.device)

    num_frames = flow_verts.shape[1]
    # Use wider gaps for larger flow: gaps of 1, 2, and 4 frames
    pairs = []
    for gap in [1, 2, 4]:
        for start in range(0, num_frames - gap):
            pairs.append((start, start + gap))
    # Sample up to 4 pairs
    if len(pairs) > 4:
        indices = np.random.choice(len(pairs), 4, replace=False)
        pairs = [pairs[i] for i in indices]

    total_loss = 0.0
    n_terms = 0

    for (start_f, end_f) in pairs:
        flow, mask = model.compute_flow(
            flow_verts[:, start_f, ...], flow_verts[:, end_f, ...],
            faces_tensor, data['intri'], args.img_size, args.img_size,
            return_mask=True,
        )

        for bb in range(batch_breaks.shape[0]):
            bstart = 0 if bb == 0 else batch_breaks[bb - 1]
            raw_events_one_batch = data['raw_events'][bstart:batch_breaks[bb]]
            time_breaks = data['event_breaks'].int()

            # Get events spanning the full gap [start_f, end_f)
            ev_start = 0 if start_f == 0 else time_breaks[bb, start_f - 1]
            ev_end = time_breaks[bb, min(end_f - 1, time_breaks.shape[1] - 1)]
            events_to_warp = raw_events_one_batch[ev_start:ev_end]

            if events_to_warp.shape[0] < 20:
                continue

            sharpness = model.compute_contrast_loss_v3(
                events_to_warp, flow[bb], mask[bb])
            total_loss += sharpness
            n_terms += 1

    if n_terms == 0:
        contrast_loss = torch.tensor(0.0, device=flow_verts.device, requires_grad=True)
    else:
        contrast_loss = args.contrast_loss * (-1) * total_loss / n_terms

    # Visualization (simplified — just show one pair)
    vis_data = {
        'loss': contrast_loss,
        'all_iwes': [], 'no_warp_iwes': [], 'gt_iwes': [],
        'flow': None, 'gt_flow': None,
    }
    if do_visualization and len(pairs) > 0:
        ref_frame = pairs[-1][0]
        flow = model.compute_flow(
            flow_verts[:, ref_frame, ...], flow_verts[:, ref_frame + 1, ...],
            faces_tensor, data['intri'], args.img_size, args.img_size,
        )
        for bb in range(min(batch_breaks.shape[0], 1)):
            bstart = 0 if bb == 0 else batch_breaks[bb - 1]
            raw_events_one_batch = data['raw_events'][bstart:batch_breaks[bb]]
            time_breaks = data['event_breaks'].int()
            event_start = 0 if ref_frame == 0 else time_breaks[bb, ref_frame - 1]
            events_to_warp = raw_events_one_batch[event_start:time_breaks[bb, ref_frame]]
            if events_to_warp.shape[0] > 10:
                vis_data['all_iwes'].append(model.compute_IWE(events_to_warp, flow[bb]))
                vis_data['no_warp_iwes'].append(model.compute_IWE(events_to_warp, torch.zeros_like(flow[bb])))
        vis_data['flow'] = flow

    return vis_data


def compute_contrast_interp(model, data, out, args, do_visualization):
    """Contrast loss at intermediate (120fps) timestamps between supervised keyframes.

    Decodes the NeMF at skip×num_steps timestamps (e.g., 8×8=64 for 120fps),
    computes flow between consecutive decoded meshes, and applies contrast loss
    on the events at those intermediate times. The supervised losses only apply
    at the 8 keyframe times, so there's no gradient conflict.

    The events between consecutive 120fps frames come from the raw_events,
    split by the event_breaks which mark keyframe boundaries. Within each
    keyframe interval, we subdivide events by time to match the 120fps frames.
    """

    B = data['events'].shape[0]
    L = data['events'].shape[1]  # 8 keyframes
    skip = args.skip  # 8 event frames per keyframe step
    device = data['events'].device

    # Total 120fps frames = L * skip = 64
    total_frames = L * skip
    decode_length = total_frames

    # Build decode timestamps matching the 120fps evaluation grid
    # The keyframe timestamps are at indices [0, skip, 2*skip, ..., (L-1)*skip]
    # in the 120fps grid. The NeMF was trained with decode_length=L,
    # so timestamps in [-1, 1] at the keyframes are:
    # t_key = [-1, -1+2/L, -1+4/L, ..., 1] = linspace(-1, 1, L)
    # For 120fps: t_hf = linspace(-1, 1, total_frames)
    decode_ts = torch.linspace(-1, 1, decode_length).unsqueeze(0).to(device)

    # Resize SMPL for the higher frame count
    model.resize_smpl(decode_length + 1)

    # Re-run the decode at 120fps using the same latent codes
    # We need to call model forward with override_clip_length
    with torch.cuda.amp.autocast():
        hf_out = model(data, override_clip_length=decode_length, decode_ts=decode_ts)

    hf_verts = hf_out['verts']  # [B, total_frames+1, 6890, 3]

    # Resize SMPL back for the normal forward pass
    model.resize_smpl(L + 1)

    faces_tensor = torch.tensor(model.smpl.faces.astype(int)).to(device)
    batch_breaks = np.cumsum(data['event_size'].cpu().numpy()).astype(int)

    total_loss = 0.0
    n_terms = 0

    # For each keyframe interval, get the intermediate 120fps frames
    # and apply contrast on them
    for step in range(L):
        # 120fps frame indices for this keyframe step
        hf_start = step * skip  # index into hf_verts (offset by 1 for init frame)
        # The hf_verts has init frame at [0], so 120fps frame k is at [k+1]
        # But actually hf_verts[:, 0] is init, hf_verts[:, 1] is first decoded frame...
        # With decode_length=64, hf_verts has 65 frames

        # Pick 2-3 random intermediate pairs within this step (not at keyframe boundaries)
        # Keyframe boundaries are at hf indices: 0, skip, 2*skip, ...
        # Intermediate indices within step: hf_start+1 to hf_start+skip-1
        intermediate_indices = list(range(hf_start + 1, hf_start + skip))
        if len(intermediate_indices) < 2:
            continue

        # Sample a few pairs
        n_pairs = min(2, len(intermediate_indices) - 1)
        pair_starts = sorted(np.random.choice(len(intermediate_indices) - 1, n_pairs, replace=False))

        for pi in pair_starts:
            idx_a = intermediate_indices[pi]  # 120fps index
            idx_b = intermediate_indices[pi] + 1

            # +1 because hf_verts has init frame at index 0
            flow, mask = model.compute_flow(
                hf_verts[:, idx_a + 1], hf_verts[:, idx_b + 1],
                faces_tensor, data['intri'], args.img_size, args.img_size,
                return_mask=True)

            for bb in range(batch_breaks.shape[0]):
                bstart = 0 if bb == 0 else batch_breaks[bb - 1]
                raw_events = data['raw_events'][bstart:batch_breaks[bb]]
                time_breaks = data['event_breaks'].int()

                # Events for this keyframe step
                ev_step_start = 0 if step == 0 else time_breaks[bb, step - 1].item()
                ev_step_end = time_breaks[bb, step].item()
                step_events = raw_events[ev_step_start:ev_step_end]

                if step_events.shape[0] < skip * 2:
                    continue

                # Subdivide step events into skip sub-intervals by time
                t_min = step_events[0, 2]
                t_max = step_events[-1, 2]
                t_range = t_max - t_min
                if t_range < 1e-6:
                    continue

                # Events for the sub-interval [idx_a - hf_start, idx_b - hf_start]
                # within this step's skip sub-intervals
                sub_idx = pi  # which sub-interval (0 to skip-2)
                sub_t_start = t_min + t_range * sub_idx / skip
                sub_t_end = t_min + t_range * (sub_idx + 1) / skip
                sub_mask = (step_events[:, 2] >= sub_t_start) & (step_events[:, 2] < sub_t_end)
                sub_events = step_events[sub_mask]

                if sub_events.shape[0] < 10:
                    continue

                sharpness = model.compute_contrast_loss_v3(
                    sub_events, flow[bb], mask[bb])
                total_loss += sharpness
                n_terms += 1

    if n_terms == 0:
        contrast_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        contrast_loss = args.contrast_loss * (-1) * total_loss / n_terms

    return {
        'loss': contrast_loss,
        'all_iwes': [], 'no_warp_iwes': [], 'gt_iwes': [],
        'flow': None, 'gt_flow': None,
    }


def train_step_forward(model, data, mse_func, device, args, iter_idx, plot_step):
    """Forward pass and loss computation shared by AMP and non-AMP paths."""
    out = model.forward(data)

    if args.our_data:
        assert abs(args.flow_loss) < 1e-5

    loss_dict = compute_losses(out, data, mse_func, device, args)
    mpjpe = compute_mpjpe(out['joints3d'].detach(), data['joints3d'])
    loss = (loss_dict['delta_tran'] + loss_dict['tran'] + loss_dict['theta'] +
            loss_dict['joints3d'] + loss_dict['joints2d'] + loss_dict['flow'])

    contrast_data = None
    if args.contrast_loss > 0:
        if getattr(args, 'contrast_interp', False):
            # Inter-frame contrast: decode at 120fps, apply contrast only at
            # intermediate timestamps where no GT exists
            contrast_data = compute_contrast_interp(model, data, out, args,
                                                     iter_idx % plot_step == 0)
        else:
            if args.use_amp:
                flow_verts = out['verts']
            else:
                flow_verts = out['verts'][:, 1:, ...]

            do_vis = (iter_idx % plot_step == 0)
            if getattr(args, 'contrast_v2', False):
                contrast_data = compute_contrast_v2(model, data, out, flow_verts, args, do_vis)
            else:
                contrast_data = compute_contrast(model, data, out, flow_verts, args, do_vis)
        loss += contrast_data['loss']

    return out, loss, loss_dict, mpjpe, contrast_data


def train(args):
    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32

    dm = EventHPEDataModule(
        args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, target_action=args.target_action,
        args=args,
    )
    model = EventHumanModel(args)

    dataset_train = dm.train_db
    dataset_val = dm.val_db

    val_generator = DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        collate_fn=custom_collate_fn,
    )
    train_generator = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        collate_fn=custom_collate_fn,
    )
    total_iters = len(dataset_train) // args.batch_size + 1

    mse_func = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    model = model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_start)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.lr_decay_step], gamma=args.lr_decay_rate,
    )

    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    writer = SummaryWriter('%s/%s/%s' % (args.result_dir, args.log_dir, start_time))
    print('[tensorboard] %s/%s/%s' % (args.result_dir, args.log_dir, start_time))

    # Configure which parameters to train for trajectory prediction
    if args.pred_traj:
        if args.finetune_gmp:
            model.gmp.requires_grad_(True)
        else:
            model.gmp.requires_grad_(False)
        if args.only_train_gmp:
            model.requires_grad_(False)
            model.gmp.requires_grad_(True)

    if args.model_dir is not None:
        print('[model dir] model loaded from %s' % args.model_dir)
        checkpoint = torch.load(args.model_dir, map_location=device)
        ckpt_state = checkpoint['model_state_dict']
        model_state = model.state_dict()
        filtered = {k: v for k, v in ckpt_state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        model_state.update(filtered)
        model.load_state_dict(model_state)
        if not args.reset_optimizer:
            if 'optimizer_state_dict' in checkpoint and not args.pred_traj:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    save_dir = '%s/%s/%s' % (args.result_dir, args.log_dir, start_time)
    model_dir = '%s/%s/%s/model_events_pose.pkl' % (args.result_dir, args.log_dir, start_time)

    if args.pred_traj and args.gmp_model_path:
        gmp_weights = torch.load(args.gmp_model_path, map_location=device)['model_state_dict']
        new_gmp_weights = {}
        for key in gmp_weights:
            new_key = key.replace('gmp.', '')
            if new_key in model.gmp.state_dict():
                new_gmp_weights[new_key] = gmp_weights[key]
        model.gmp.load_state_dict(new_gmp_weights)

    best_loss = 1e4
    PLOT_STEP = 100

    for epoch in range(args.epochs):
        print('=' * 30 + ' Epoch %i ' % (epoch + 1) + '=' * 30)

        # ---- Training ----
        print('-' * 30 + ' Training ' + '-' * 30)
        model.train()
        results = collections.defaultdict(list)
        start_time = time.time()

        for iter, data in enumerate(tqdm(train_generator)):
            for k in data:
                if k != 'info':
                    data[k] = data[k].to(device=device, dtype=dtype)

            optimizer.zero_grad()

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    out, loss, loss_dict, mpjpe, contrast_data = train_step_forward(
                        model, data, mse_func, device, args, iter, PLOT_STEP,
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out, loss, loss_dict, mpjpe, contrast_data = train_step_forward(
                    model, data, mse_func, device, args, iter, PLOT_STEP,
                )
                loss.backward()
                optimizer.step()

            # Collect scalar results
            results['scalar/delta_tran'].append(loss_dict['delta_tran'].detach())
            results['scalar/tran'].append(loss_dict['tran'].detach())
            results['scalar/theta'].append(loss_dict['theta'].detach())
            results['scalar/joints3d'].append(loss_dict['joints3d'].detach())
            results['scalar/joints2d'].append(loss_dict['joints2d'].detach())
            results['scalar/flow'].append(loss_dict['flow'].detach())
            results['scalar/loss'].append(loss.detach())
            results['scalar/mpjpe'].append(torch.mean(mpjpe.detach()))

            if contrast_data is not None:
                results['scalar/contrast_loss'].append(contrast_data['loss'].detach())

            for jid in range(24):
                results['scalar/debug_theta_%i' % jid].append(loss_dict['theta_%i' % jid].detach())

            # Mid-epoch checkpoint save
            if iter > 0 and iter % 5000 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, '%s/model_iter_%d.pkl' % (save_dir, iter))
                print('>>> Mid-epoch checkpoint saved at iter %d' % iter)

            # Mid-epoch validation
            if args.val_every > 0 and iter > 0 and iter % args.val_every == 0:
                model.eval()
                val_results = collections.defaultdict(list)
                with torch.set_grad_enabled(False):
                    for vdata in tqdm(val_generator, desc='mid-val'):
                        for k in vdata:
                            if k != 'info':
                                vdata[k] = vdata[k].to(device=device, dtype=dtype)
                        vout = model(vdata)
                        vmpjpe = compute_mpjpe(vout['joints3d'].detach(), vdata['joints3d'])
                        vpa = compute_pa_mpjpe(vout['joints3d'].detach(), vdata['joints3d'])
                        vpel = compute_pelvis_mpjpe(vout['joints3d'].detach(), vdata['joints3d'])
                        vpck = compute_pck(vout['joints3d'], vdata['joints3d'])
                        val_results['mpjpe'].append(torch.mean(vmpjpe.detach()))
                        val_results['pa_mpjpe'].append(torch.mean(vpa.detach()))
                        val_results['pel_mpjpe'].append(torch.mean(vpel.detach()))
                        val_results['pck'].append(torch.mean(vpck.detach().float(), dim=(0, 1, 2)))
                vr = {k: torch.mean(torch.stack(v)) for k, v in val_results.items()}
                print('>>> [mid-val iter %d] mpjpe %.4f, pa_mpjpe %.4f, pel_mpjpe %.4f, pck %.4f'
                      % (iter, 1000*vr['mpjpe'], 1000*vr['pa_mpjpe'], 1000*vr['pel_mpjpe'], vr['pck']))
                writer.add_scalar('val/mpjpe', vr['mpjpe']*1000, epoch*total_iters+iter)
                writer.add_scalar('val/pa_mpjpe', vr['pa_mpjpe']*1000, epoch*total_iters+iter)
                writer.add_scalar('val/pel_mpjpe', vr['pel_mpjpe']*1000, epoch*total_iters+iter)
                writer.add_scalar('val/pck', vr['pck'], epoch*total_iters+iter)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, '%s/model_iter_%d.pkl' % (save_dir, iter))
                model.train()
                if args.pred_traj:
                    if not args.finetune_gmp:
                        model.gmp.requires_grad_(False)
                    if args.only_train_gmp:
                        model.requires_grad_(False)
                        model.gmp.requires_grad_(True)

            # Periodic logging
            display = 200
            if iter % (total_iters // display) == 0:
                results['info'] = (data['info'][0][0], data['info'][1][0])
                results['verts'] = out['verts'][0].detach()

                for key_name in ['delta_tran', 'tran', 'theta', 'joints3d', 'joints2d',
                                 'flow', 'loss', 'mpjpe']:
                    results[key_name] = torch.mean(torch.stack(results['scalar/%s' % key_name], dim=0))

                if args.contrast_loss > 0:
                    results['contrast_loss'] = torch.mean(
                        torch.stack(results['scalar/contrast_loss'], dim=0))

                for jid in range(24):
                    results['debug_theta_%i' % jid] = torch.mean(
                        torch.stack(results['scalar/debug_theta_%i' % jid], dim=0))

                progress = (100 // display) * iter // (total_iters // display) + 1

                for key in results:
                    if isinstance(results[key], torch.Tensor) and results[key].dim() == 0:
                        if 'debug' in key:
                            writer.add_scalar('debug/%s' % key, results[key].detach(),
                                              epoch * total_iters + iter)
                        elif 'mpjpe' in key:
                            writer.add_scalar('train/%s' % key, results[key].detach() * 1000,
                                              epoch * total_iters + iter)
                        else:
                            writer.add_scalar('train/%s' % key, results[key].detach(),
                                              epoch * total_iters + iter)

                end_time = time.time()
                time_used = (end_time - start_time) / 60.
                print('>>> [epoch {:2d}/ iter {:6d}] {:3d}%\n'
                      '    loss {:.4f}, tran {:.4f}, theta {:.4f}, '
                      'joints3d {:.4f}, joints2d {:.4f} \n'
                      '    flow {:.4f}, delta_tran {:.4f}, mpjpe {:.4f} mm \n'
                      '    lr: {:.6f}, time used: {:.2f} mins, '
                      'still need time for this epoch: {:.2f} mins.'
                      .format(epoch, iter, progress - 1,
                              results['loss'], results['tran'], results['theta'],
                              results['joints3d'], results['joints2d'],
                              results['flow'], results['delta_tran'],
                              1000 * results['mpjpe'],
                              scheduler.get_last_lr()[0], time_used,
                              (100 / progress - 1) * time_used))

            # Periodic visualization
            if iter % PLOT_STEP == 0:
                pred_images = model.plot_human(data, out)
                writer.add_images('train/pred_images', pred_images,
                                  epoch * total_iters + iter, dataformats='NCHW')

                if args.contrast_loss > 0 and contrast_data is not None:
                    # Visualize IWE from a consistent frame pair (frame 3->4)
                    vis_verts = out['verts']
                    vis_ref = min(3, vis_verts.shape[1] - 2)
                    faces_t = torch.tensor(model.smpl.faces.astype(int)).to(vis_verts.device)
                    vis_flow, vis_mask = model.compute_flow(
                        vis_verts[:1, vis_ref], vis_verts[:1, vis_ref + 1],
                        faces_t, data['intri'][:1], args.img_size, args.img_size,
                        return_mask=True)

                    # Get events for this same frame pair
                    vis_breaks = data['event_breaks'].int()
                    vis_estart = 0 if vis_ref == 0 else vis_breaks[0, vis_ref - 1].item()
                    vis_eend = vis_breaks[0, vis_ref].item()
                    vis_events = data['raw_events'][0, vis_estart:vis_eend]

                    if vis_events.shape[0] > 10:
                        iwe_w = model.compute_IWE(vis_events, vis_flow[0])
                        iwe_nw = model.compute_IWE(vis_events, torch.zeros_like(vis_flow[0]))
                        sil_vis = vis_mask[0]

                        iwe_w = iwe_w / iwe_w.max().clamp(min=1e-6)
                        iwe_nw = iwe_nw / iwe_nw.max().clamp(min=1e-6)
                        diff = (iwe_w - iwe_nw).abs()
                        diff = diff / diff.max().clamp(min=1e-6)

                        imgs = []
                        for img in [iwe_nw, iwe_w, diff, sil_vis]:
                            imgs.append(img[None, ...].repeat(3, 1, 1))
                        flow_img = flow_to_image(vis_flow[:1].permute(0, 3, 1, 2)).float() / 255.
                        imgs.append(flow_img[0])

                        vis = torch.stack(imgs, dim=0)
                        writer.add_images('train/contrast_iwe', vis,
                                          epoch * total_iters + iter, dataformats='NCHW')

                        var_w = torch.var(iwe_w * sil_vis).item() if sil_vis.sum() > 0 else 0
                        var_nw = torch.var(iwe_nw * sil_vis).item() if sil_vis.sum() > 0 else 0
                        writer.add_scalar('train/iwe_var_warped', var_w, epoch * total_iters + iter)
                        writer.add_scalar('train/iwe_var_nowarped', var_nw, epoch * total_iters + iter)
                        writer.add_scalar('train/iwe_var_ratio', var_w / max(var_nw, 1e-6),
                                          epoch * total_iters + iter)

        # ---- Validation ----
        print('-' * 30 + ' test ' + '-' * 30)
        start_time = time.time()
        model.eval()
        results = collections.defaultdict(list)

        with torch.set_grad_enabled(False):
            for iter, data in enumerate(tqdm(val_generator)):
                for k in data:
                    if k != 'info':
                        data[k] = data[k].to(device=device, dtype=dtype)

                out = model(data)
                loss_dict = compute_losses(out, data, mse_func, device, args)
                mpjpe = compute_mpjpe(out['joints3d'].detach(), data['joints3d'])
                pa_mpjpe = compute_pa_mpjpe(out['joints3d'].detach(), data['joints3d'])
                pel_mpjpe = compute_pelvis_mpjpe(out['joints3d'].detach(), data['joints3d'])
                pck = compute_pck(out['joints3d'], data['joints3d'])

                loss = (loss_dict['delta_tran'] + loss_dict['tran'] + loss_dict['theta'] +
                        loss_dict['joints3d'] + loss_dict['joints2d'] + loss_dict['flow'])

                results['scalar/delta_tran'].append(loss_dict['delta_tran'].detach())
                results['scalar/tran'].append(loss_dict['tran'].detach())
                results['scalar/theta'].append(loss_dict['theta'].detach())
                results['scalar/joints3d'].append(loss_dict['joints3d'].detach())
                results['scalar/joints2d'].append(loss_dict['joints2d'].detach())
                results['scalar/flow'].append(loss_dict['flow'].detach())
                results['scalar/loss'].append(loss.detach())
                results['scalar/mpjpe'].append(torch.mean(mpjpe.detach()))
                results['scalar/pa_mpjpe'].append(torch.mean(pa_mpjpe.detach()))
                results['scalar/pel_mpjpe'].append(torch.mean(pel_mpjpe.detach()))
                results['scalar/pck'].append(torch.mean(pck.detach().float(), dim=(0, 1, 2)))

                if iter % PLOT_STEP == 0:
                    results['info'] = (data['info'][0][0], data['info'][1][0])
                    results['verts'] = out['verts'][0].detach()
                    pred_images = model.plot_human(data, out)
                    writer.add_images('test/pred_images', pred_images,
                                      epoch * total_iters + iter, dataformats='NCHW')

            for key_name in ['delta_tran', 'tran', 'theta', 'joints3d', 'joints2d',
                             'flow', 'loss', 'mpjpe', 'pa_mpjpe', 'pel_mpjpe', 'pck']:
                results[key_name] = torch.mean(
                    torch.stack(results['scalar/%s' % key_name], dim=0))

            end_time = time.time()
            time_used = (end_time - start_time) / 60.
            print('>>> loss {:.4f}, tran {:.4f}, theta {:.4f}, '
                  'joints3d {:.4f}, joints2d {:.4f} \n'
                  '    flow {:.4f}, delta_tran {:.4f}, time used: {:.2f} mins \n'
                  '    mpjpe {:.4f}, pa_mpjpe {:.4f}, pel_mpjpe {:.4f} pck {:.4f}'
                  .format(results['loss'], results['tran'], results['theta'],
                          results['joints3d'], results['joints2d'],
                          results['flow'], results['delta_tran'], time_used,
                          1000 * results['mpjpe'], 1000 * results['pa_mpjpe'],
                          1000 * results['pel_mpjpe'], results['pck']))

            for key in results:
                if isinstance(results[key], torch.Tensor) and results[key].dim() == 0:
                    if 'mpjpe' in key:
                        writer.add_scalar('val/%s' % key, results[key].detach() * 1000,
                                          epoch * total_iters + iter)
                    else:
                        writer.add_scalar('val/%s' % key, results[key].detach(),
                                          epoch * total_iters + iter)

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, '%s/model.pkl' % save_dir)

            if best_loss > results['loss']:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, model_dir)
                best_loss = results['loss']
                print('>>> Model saved as {}... best loss {:.4f}'.format(model_dir, best_loss))

        scheduler.step()
    writer.close()


def get_args():
    def print_args(args):
        _args = vars(args)
        max_length = max([len(k) for k, _ in _args.items()])
        for k, v in _args.items():
            print(' ' * (max_length - len(k)) + k + ': ' + str(v))

    from NeMF.src.arguments import Arguments

    args = Arguments('NeMF/configs', filename="application.yaml")
    args.config_path = 'NeMF/configs'

    parser = argparse.ArgumentParser()

    # Paths and environment
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--data_root', type=str, default='data/mmhpsd')
    parser.add_argument('--result_dir', type=str, default='outputs')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--smpl_dir', type=str, default='event_hpe/smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
    parser.add_argument('--event_folder', type=str, default='data/mmhpsd_events/')
    parser.add_argument('--gmp_model_path', type=str, default=None)

    # Data
    parser.add_argument('--target_action', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=int, default=1)
    parser.add_argument('--clip_len', type=int, default=8)
    parser.add_argument('--num_event_channels', type=int, default=8)
    parser.add_argument('--events_input_channel', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--max_steps', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=8)
    parser.add_argument('--skip', type=int, default=2)

    # Model
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--use_hmr_feats', type=int, default=1)
    parser.add_argument('--use_flow', type=int, default=1)
    parser.add_argument('--use_geodesic_loss', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet34')
    parser.add_argument('--hmr_model', action='store_true')
    parser.add_argument('--ours_full_pose0', action='store_true')
    parser.add_argument('--left_mult', action='store_true')
    parser.add_argument('--abl_transformer', action='store_true')
    parser.add_argument('--use_volumes', action='store_true')
    parser.add_argument('--no_pose0', action='store_true')
    parser.add_argument('--our_data', action='store_true')

    # Training
    parser.add_argument('--use_amp', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_start', '-lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_step', type=float, default=1)
    parser.add_argument('--reset_optimizer', action='store_true')
    parser.add_argument('--val_every', type=int, default=0,
                        help='Run validation every N training iters (0=end of epoch only)')
    parser.add_argument('--contrast_v2', action='store_true',
                        help='Use improved contrast loss (multi-pair, polarity split, silhouette mask)')
    parser.add_argument('--use_h5', action='store_true',
                        help='Use pre-packed H5 data loader for faster training')
    parser.add_argument('--contrast_interp', action='store_true',
                        help='Apply contrast loss at intermediate 120fps timestamps only')

    # Loss weights
    parser.add_argument('--delta_tran_loss', type=float, default=0)
    parser.add_argument('--tran_loss', type=float, default=1)
    parser.add_argument('--theta_loss', type=float, default=10)
    parser.add_argument('--joints3d_loss', type=float, default=1)
    parser.add_argument('--joints2d_loss', type=float, default=10)
    parser.add_argument('--flow_loss', type=float, default=0.1)
    parser.add_argument('--contrast_loss', type=float, default=0)

    # Trajectory prediction
    parser.add_argument('--pred_traj', action='store_true')
    parser.add_argument('--finetune_gmp', action='store_true')
    parser.add_argument('--only_train_gmp', action='store_true')

    # Testing
    parser.add_argument('--test_high_fps', action='store_true')

    my_args = parser.parse_args()
    for k, v in vars(my_args).items():
        setattr(args, k, v)

    print_args(args)
    return args


def main():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    train(args)


if __name__ == '__main__':
    main()
