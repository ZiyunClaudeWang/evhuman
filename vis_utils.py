import numpy as np
import torch

def kp_crop_to_original(keypoints, center, scale, crop_size=256):
    '''
    keypoints: (B, K, 3) tensor
    center: (B, 2) tensor
    scale: (B,) tensor
    '''
    hs = crop_size / 2
    keypoints_xy = keypoints[:, :, :2] * hs + hs
    b = scale * 200
    uncropped_v = keypoints_xy * (b[:, None, None] / 256) \
                + (center[:, None, :] - b[:, None, None]/2)
    return torch.cat([uncropped_v, keypoints[:, :, 2:]], dim=2)

def kp_crop_to_crop_pixel(keypoints, center, scale, crop_size=256):
    hs = crop_size / 2
    keypoints_xy = keypoints[:, :, :2] * hs + hs
    return keypoints_xy

def pix_crop_to_original(pixel_xy, center, scale, crop_size=256):
    '''
    keypoints: (B, K, 2) tensor
    center: (B, 2) tensor
    scale: (B,) tensor
    '''
    b = scale * 200
    uncropped_v = pixel_xy * (b[:, None, None] / 256) \
                + (center[:, None, :] - b[:, None, None]/2)
    return uncropped_v 

def pix_orignal_to_crop(pixel_xy, center, scale, crop_size=256):
    '''
    keypoints: (B, K, 2) tensor
    center: (B, 2) tensor
    scale: (B,) tensor
    '''
    hs = crop_size / 2
    b = scale * 200
    cropped_v = (pixel_xy[:, :, :2] - (center[:, None, :] - b[:, None, None]/2)) \
                * (256 / b[:, None, None])
    return cropped_v


def gen_event_images(event_volume, prefix, device="cuda", clamp_val=2., normalize_events=True, signed=True):
    
    # if signed:
    #     n_bins = int(event_volume.shape[1] / 2)
    #     time_range = torch.tensor(np.linspace(0.1, 1, n_bins), dtype=torch.float32).to(device)
    #     time_range = torch.reshape(time_range, (1, n_bins, 1, 1))
        
    #     pos_event_image = torch.sum(
    #         event_volume[:, :n_bins, ...] * time_range / \
    #         (torch.sum(event_volume[:, :n_bins, ...], dim=1, keepdim=True) + 1e-5),
    #         dim=1, keepdim=True)
    #     neg_event_image = torch.sum(
    #         event_volume[:, n_bins:, ...] * time_range / \
    #         (torch.sum(event_volume[:, n_bins:, ...], dim=1, keepdim=True) + 1e-5),
    #         dim=1, keepdim=True)
    #     event_time_image = (pos_event_image + neg_event_image) / 2.
    # else:
    n_bins = event_volume.shape[1]
    time_range = torch.tensor(np.linspace(0.1, 1, n_bins), dtype=torch.float32).to(device)
    time_range = torch.reshape(time_range, (1, n_bins, 1, 1))

    event_time_image = torch.sum(
            event_volume[:, :, ...] * time_range / \
            (torch.sum(event_volume[:, :, ...], dim=1, keepdim=True) + 1e-5),
        dim=1, keepdim=True)

    return event_time_image