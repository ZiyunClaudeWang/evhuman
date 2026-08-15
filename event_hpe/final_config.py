"""Canonical model configuration of the released method.

The research code accumulated experimental switches (ablations, alternative
regressors, initialization variants). The released method corresponds to one
fixed setting of all of them; exposing the switches only invites
misconfiguration. Every entry-point script calls apply_final_config() after
parsing its arguments, so the internals are pinned here and the parsers keep
only genuinely user-facing options (paths, dataset, batch size, losses,
schedule).
"""

FINAL_CONFIG = {
    # architecture of the released model
    'hmr_model': True,          # ResNet event encoder + iterative regressor
    'ours_full_pose0': True,    # root orientation grafted from the init pose
    'pred_traj': True,          # GMP predicts root translation
    'backbone': 'resnet34',
    'rnn_layers': 1,
    'img_size': 256,
    'clip_len': 8,
    'num_event_channels': 8,
    'events_input_channel': 8,
    'max_steps': 8,
    'num_steps': 8,
    'use_flow': 1,
    'use_hmr_feats': 0,
    'use_geodesic_loss': 1,

    # ablations / abandoned variants — permanently off
    'abl_transformer': False,
    'no_pose0': False,
    'left_mult': True,          # legacy flag, no code reads it
    'use_volumes': False,
    'hmr_interp': False,
    'no_pretrained': False,
    'direct_regress': False,
    'regress_6d': False,
    'use_flow_rgb': 0,
    'vibe_regressor': 0,
    'use_vibe_init': 0,
    'use_hmr_init': 0,
    'use_gt_trans': False,
    'use_gt_root': False,
    'latent_trans': False,
    'use_frames': False,
}


def apply_final_config(args):
    """Pin all internal switches to the released configuration."""
    for key, value in FINAL_CONFIG.items():
        setattr(args, key, value)
    return args
