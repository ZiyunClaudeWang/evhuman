import os
import h5py
import numpy as np
import cv2
from torch.utils.data import Dataset
import pickle
import joblib
import torch
# from flowlib import flow_to_image

class OursTrackingDataloader(Dataset):
    def __init__(
            self,
            data_dir='data/beahm',
            max_steps=16,
            num_steps=8,
            skip=2,
            events_input_channel=8,
            img_size=256,
            mode='train',
            use_flow=True,
            use_flow_rgb=False,
            use_hmr_feats=False,
            use_vibe_init=False,
            use_hmr_init=False,
            target_action=None,
            raw_events=False,
            event_folder="data/beahm_events/",
            use_volumes=False,
            test_high_fps=False
    ):
        self.data_dir = data_dir
        self.events_input_channel = events_input_channel
        self.skip = skip
        self.max_steps = max_steps
        self.num_steps = num_steps
        self.img_size = img_size
        self.scale = self.img_size / 640.
        self.use_hmr_feats = use_hmr_feats
        self.use_flow = use_flow
        self.use_flow_rgb = use_flow_rgb
        self.use_vibe_init = use_vibe_init
        self.use_hmr_init = use_hmr_init
        self.raw_events = raw_events
        self.event_folder = event_folder

        self.use_volumes = use_volumes
        self.test_high_fps = test_high_fps

        self.mode = mode
        if os.path.exists('%s/%s_track%02i%02i.pkl' % (self.data_dir, self.mode, self.num_steps, self.skip)):
            self.all_clips = pickle.load(
                open('%s/%s_track%02i%02i.pkl' % (self.data_dir, self.mode, self.num_steps, self.skip), 'rb'))
        else:
            self.all_clips = self.obtain_all_clips()

        if self.use_vibe_init:
            print('[VIBE init]')
            all_clips = []
            for (action, frame_idx) in self.all_clips:
                if os.path.exists('%s/vibe_results_%02i%02i/%s/fullpic%04i_vibe%02i.pkl' %
                                  (self.data_dir, self.num_steps, self.skip, action, frame_idx, self.num_steps)):
                    all_clips.append((action, frame_idx))
                else:
                    print('[vibe not exist] %s %i' % (action, frame_idx))
            self.all_clips = all_clips

        if self.use_hmr_init:
            print('[hmr init]')
            all_clips = []
            for (action, frame_idx) in self.all_clips:
                if os.path.exists('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx)):
                    all_clips.append((action, frame_idx))
                else:
                    print('[hmr not exist] %s %i' % (action, frame_idx))
            self.all_clips = all_clips

        if target_action:
            clips = self.all_clips
            good_clips = []
            for (action, frame_idx) in clips:
                if target_action in action:
                    good_clips.append((action, frame_idx))
            self.all_clips = good_clips
        

        print('[%s] %i clips, track%02i%02i.pkl' % (self.mode, len(self.all_clips), self.num_steps, self.skip))

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):
        action, frame_idx = self.all_clips[idx]
        if self.mode == 'train':
            next_frames_idx = self.skip * np.sort(np.random.choice(
                np.arange(1, self.max_steps+1), self.num_steps, replace=False))
        else:
            # test
            next_frames_idx = self.skip * np.arange(1, self.num_steps+1)

        sample_frames_idx = np.append(frame_idx, frame_idx + next_frames_idx)

        if self.use_vibe_init:
            _, _, _params, _tran = joblib.load(
                '%s/vibe_results_%02i%02i/%s/fullpic%04i_vibe%02i.pkl' %
                (self.data_dir, self.num_steps, self.skip, action, frame_idx, self.num_steps))
            theta = _params[0:1, 3:75]
            beta = _params[0:1, 75:]
            tran = _tran[0:1, :]
            init_shape = np.concatenate([tran, theta, beta], axis=1)
        elif self.use_hmr_init:
            _, _, _params, _tran, _ = \
                joblib.load('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx))
            theta = np.expand_dims(_params[3:75], axis=0)
            beta = np.expand_dims(_params[75:], axis=0)
            tran = _tran
            init_shape = np.concatenate([tran, theta, beta], axis=1)
        else:
            beta, theta, tran, _, _, _ = joblib.load('%s/pose_events/%s/pose%04i.pkl' % (self.data_dir, action, frame_idx))
            
            init_shape = np.concatenate([tran.reshape(1,-1), theta.reshape(1,-1), beta.reshape(1,-1)], axis=1)

        if self.use_hmr_feats:
            _, _, _, _, hmr_feats = joblib.load(
                '%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx))  # [2048]
        else:
            hmr_feats = np.zeros([2048])

        events, flows, flows_rgb, theta_list, tran_list, joints2d_list, joints3d_list = [], [], [], [], [], [], []

        high_fps_joints3d_list = []
        high_fps_joints2d_list = []
        high_fps_theta_list = []
        high_fps_tran_list = []

        for i in range(self.num_steps):
            start_idx = sample_frames_idx[i]
            end_idx = sample_frames_idx[i+1]
            # print('frame %i - %i' % (start_idx, end_idx))

            # single step events frame
            single_events_frame = []
            for j in range(start_idx, end_idx):
                if self.use_volumes:
                    volume_dir = os.path.join(os.path.dirname(self.data_dir), "beahm_volumes")
                    loaded_event_frame = cv2.imread('%s/%s/event%04i.png' % (volume_dir, action, j), -1)
                    loaded_event_frame = cv2.resize(loaded_event_frame, (self.img_size, self.img_size))
                    single_events_frame.append(loaded_event_frame)

                else:
                    single_events_frame.append(cv2.imread(
                        '%s/events_%i/%s/event%04i.png' % (self.data_dir, self.img_size, action, j), -1))

            single_events_frame = np.concatenate(single_events_frame, axis=2).astype(np.float32)  # [H, W, C]
            # aggregate the events frame to get 8 channel
            if single_events_frame.shape[2] > self.events_input_channel:
                skip = single_events_frame.shape[2] // self.events_input_channel
                idx1 = skip * np.arange(self.events_input_channel)
                idx2 = idx1 + skip
                idx2[-1] = max(idx2[-1], single_events_frame.shape[2])
                single_events_frame = np.stack(
                    [(np.sum(single_events_frame[:, :, c1:c2], axis=2) > 0) for (c1, c2) in zip(idx1, idx2)], axis=2)
            events.append(single_events_frame)

            single_flows = np.zeros([2, self.img_size, self.img_size])
            flows.append(single_flows)
            # flows_rgb.append(single_flows_rgb)

            # single frame pose
            if self.test_high_fps:
                for idd in range(start_idx+1, end_idx + 1):
                    beta, theta, tran, joints3d, joints2d, intri = joblib.load(
                        '%s/pose_events/%s/pose%04i.pkl' % (self.data_dir, action, idd))
                    high_fps_theta_list.append(theta)
                    high_fps_tran_list.append(tran)
                    high_fps_joints2d_list.append(joints2d)
                    high_fps_joints3d_list.append(joints3d)

                # _, _, _, next_joints3d, _, _ = joblib.load(
                #     '%s/pose_events/%s/pose%04i.pkl' % (self.data_dir, action, end_idx+1))

            beta, theta, tran, joints3d, joints2d, intri = joblib.load(
                '%s/pose_events/%s/pose%04i.pkl' % (self.data_dir, action, end_idx))
            theta_list.append(theta)
            tran_list.append(tran)
            joints2d_list.append(joints2d)
            joints3d_list.append(joints3d)
        
        # get events directly
        if self.raw_events:
            raw_event_file = os.path.join(self.event_folder, '{}.h5'.format(action))
            img2events_file = os.path.join(self.event_folder, '{}_img2events.npy'.format(action))
            # if os.path.exists(img2events_file):
            #     img2events = np.load(img2events_file)
            # else:
            with h5py.File(raw_event_file, 'r') as f:
                img_time = f['events']['event_annot_ts'][:]
                events_time = f['events']['t'][:]
                img2events = np.searchsorted(events_time, img_time)
                np.save(img2events_file, img2events) 
            
            with h5py.File(raw_event_file, 'r') as f:
                # image_in_events = f['images']['image_annot_ts'][:]
                image_in_events = img2events


                sample_start_idx = image_in_events[sample_frames_idx[0]]
                # add one frame to have events at the end

                if sample_frames_idx[-1] > image_in_events.shape[0] - 1:
                    sample_end_idx = image_in_events[-1]
                else:
                    sample_end_idx = image_in_events[sample_frames_idx[-1]]

                # get events between first to the last
                yy = f['events']['x'][sample_start_idx: sample_end_idx][:]
                xx = f['events']['y'][sample_start_idx: sample_end_idx][:]

                points = np.stack([xx, yy], axis=1).astype(np.float64)[:, None, :]

                intr = f['calibration/event_intr'][:]
                dist = f['calibration/event_dist'][:]

                undist_points = undist_point(points, intr, dist, None, 0.6)

                xx = undist_points[0, :]
                yy = undist_points[1, :]

                raw_x = (xx + 80.) * 256. / 640
                raw_y = yy * 256. / 640

                raw_p = f['events']['p'][sample_start_idx: sample_end_idx][:]
                raw_t = f['events']['t'][sample_start_idx: sample_end_idx][:]

                raw_events = np.stack([raw_x, raw_y, raw_t, raw_p], axis=1)

                if self.raw_events:
                    # event_break = np.append(image_in_events[sample_frames_idx])
                    event_break = image_in_events[sample_frames_idx]
                    event_breaks = event_break[1:] - event_break[0]

        events = np.stack(events, axis=0)  # [T, H, W, 8]
        flows = np.stack(flows, axis=0)  # [T, 2/3, H, W]
        theta_list = np.stack(theta_list, axis=0)  # [T, 72]
        tran_list = np.expand_dims(np.stack(tran_list, axis=0), axis=1)  # [T, 1, 3] in meter
        # [T, 24, 2] drop d, normalize to 0-1
        joints2d_list = np.stack(joints2d_list, axis=0)[:, :, 0:2] / self.img_size
        joints3d_list = np.stack(joints3d_list, axis=0)  # [T, 24, 3] added trans


        one_sample = {}
        one_sample['events'] = torch.from_numpy(np.transpose(events, [0, 3, 1, 2])).float()  # [T, 8, H, W]
        one_sample['flows'] = torch.from_numpy(flows).float()  # [T, 2, H, W]

        if self.raw_events:
            one_sample['raw_events'] = torch.from_numpy(raw_events).float()  # [N, 4]
            one_sample['event_size'] = raw_events.shape[0]
            one_sample['event_breaks'] = event_breaks

        # one_sample['flows_rgb'] = torch.from_numpy(flows).float()  # [T, 3, H, W]
        one_sample['init_shape'] = torch.from_numpy(init_shape).float()  # [1, 85]
        one_sample['hidden_feats'] = torch.from_numpy(hmr_feats).float()  # [2048]
        one_sample['theta'] = torch.from_numpy(theta_list).float()  # [T, 72]
        one_sample['tran'] = torch.from_numpy(tran_list).float()  # [T, 1, 3]
        one_sample['joints2d'] = torch.from_numpy(joints2d_list).float()  # [T, 24, 2]
        one_sample['joints3d'] = torch.from_numpy(joints3d_list).float()  # [T, 24, 3]
        one_sample['info'] = [action, sample_frames_idx]
        one_sample['intri'] = intri

        if self.test_high_fps:
            high_fps_theta_list = np.stack(high_fps_theta_list, axis=0)
            high_fps_tran_list = np.expand_dims(np.stack(high_fps_tran_list, axis=0), axis=1)
            high_fps_joints2d_list = np.stack(high_fps_joints2d_list, axis=0)[:, :, 0:2] / self.img_size
            high_fps_joints3d_list = np.stack(high_fps_joints3d_list, axis=0)
            one_sample['high_fps_theta'] = torch.from_numpy(high_fps_theta_list).float()  # [T, 72]
            one_sample['high_fps_tran'] = torch.from_numpy(high_fps_tran_list).float()  # [T, 1, 3]
            one_sample['high_fps_joints2d'] = torch.from_numpy(high_fps_joints2d_list).float()  # [T, 24, 2]
            one_sample['high_fps_joints3d'] = torch.from_numpy(high_fps_joints3d_list).float()  # [T, 24, 3]
            # one_sample['next_joints3d'] = torch.from_numpy(next_joints3d).float()  # [24, 3]

        return one_sample

    def obtain_all_clips(self):
        all_clips = []
        tmp = sorted(os.listdir('%s/pose_events' % self.data_dir))
        action_names = []
        test_motions = ['Squat', 'Starjump', 'Walking', 'Jogging',
                        'Jumpforback', 'Jumpsideway', 'Jumpupdown',
                        'Leanleft', 'Leanright']
        for action in tmp:
            motion = action.split('_')[0]
            subject = action.split('_')[1]
            idx = action.split('_')[-1]
            # the public BEAHM release anonymizes subjects (rex -> n1,
            # ziyan -> n2) and drops the capture date; capture indices are
            # preserved, so the split is identical under both namings
            is_test = (motion in test_motions
                       and subject in ['ziyan', 'n2']
                       and idx in ['1'])
            if self.mode == 'test':
                if is_test:
                    action_names.append(action)
            else:
                if not is_test:
                    action_names.append(action)

        for action in action_names:
            if not os.path.exists('%s/pose_events/%s/pose_info.pkl' % (self.data_dir, action)):
                print('[warning] not exsit %s/pose_events/%s/pose_info.pkl' % (self.data_dir, action))
                continue

            frame_indices = joblib.load('%s/pose_events/%s/pose_info.pkl' % (self.data_dir, action))
            for i in range(len(frame_indices) - self.max_steps * self.skip):
                frame_idx = frame_indices[i]
                end_frame_idx = frame_idx + self.max_steps * self.skip
                # if not os.path.exists('%s/pred_flow_events_%i/%s/flow%04i.pkl' %
                #                       (self.data_dir, self.img_size, action, end_frame_idx)):
                #     # print('flow %i not exists for %s-%i' % (end_frame_idx, action, frame_idx))
                #     continue
                # if not os.path.exists(
                #         '%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx)):
                #     continue
                if end_frame_idx == frame_indices[i + self.max_steps * self.skip]:
                    # action, frame_idx
                    all_clips.append((action, frame_idx))
                # else:
                #     print(end_frame_idx-frame_indices[i + self.max_steps * self.skip])
        print(len(all_clips))
        pickle.dump(all_clips, open('%s/%s_track%02i%02i.pkl' %
                                    (self.data_dir, self.mode, self.num_steps, self.skip), 'wb'))
        return all_clips


def undist_point(keypoints, proj, dist, size, cooef): # keypoints in (3,N)
    _xi = np.array([proj[0]])
    _recip_fu = proj[1]
    _recip_fv = proj[2]
    _cu = proj[3]
    _cv = proj[4]
    K = np.array([[_recip_fu, 0, _cu],
                    [0, _recip_fv, _cv],
                    [0, 0, 1.0]])
    P = np.array([[_recip_fu*cooef, 0, _cu],
                    [0, _recip_fv*cooef, _cv],
                    [0, 0, 1.0]])
    
    point3d = cv2.omnidir.undistortPoints(keypoints, K, dist, _xi, np.eye(3)) 
    point3d = point3d.reshape(-1,2)
    point3d = np.hstack([point3d,np.ones((point3d.shape[0],1))])
    proj = P @ point3d.T 
    proj = proj[:2] / proj[2]
    return proj
    # def visualize(self, idx):
    #     sample = self.__getitem__(idx)
    #     action, sample_frames_idx = sample['info']
    #     events = np.transpose(sample['events'].numpy(), [0, 2, 3, 1])
    #     flows = np.transpose(sample['flows'].numpy(), [0, 2, 3, 1])
    #     joints3d = sample['joints3d'].numpy()
    #     joints2d = sample['joints2d'].numpy()* self.img_size 
    #     intri = sample['intri']

    #     # '''
    #     from SMPL import SMPL
    #     model_dir = './event_pose_estimation/thirdparty/smpl_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    #     device = torch.device('cpu')
    #     smpl_male = SMPL(model_dir, 1).to(device)

    #     # import utils as util
    #     import matplotlib.pyplot as plt
    #     from flowlib import flow_to_image
    #     from geometry import projection_torch

    #     for t in range(self.num_steps // 2):
    #         plt.figure(figsize=(5 * 4, 5 * 4))

    #         # fullpic
    #         prev_fullpic = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.png' %
    #                                   (self.data_dir, self.img_size, action, sample_frames_idx[t]))
    #         curr_fullpic = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.png' %
    #                                   (self.data_dir, self.img_size, action, sample_frames_idx[t+1]))
    #         plt.subplot(4, 4, 1)
    #         plt.imshow(prev_fullpic[:, :, 0], cmap='gray')
    #         plt.axis('off')
    #         plt.title('%s-%04i' % (action, sample_frames_idx[t]))

    #         plt.subplot(4, 4, 2)
    #         plt.imshow(curr_fullpic[:, :, 0], cmap='gray')
    #         plt.axis('off')
    #         plt.title('%s-%04i' % (action, sample_frames_idx[t+1]))

    #         for i in range(self.num_steps):
    #             plt.subplot(4, 4, 3+i)
    #             plt.imshow(events[t, :, :, i], cmap='gray')
    #             plt.axis('off')

    #         beta = sample['init_shape'][:, -10:]
    #         theta = sample['theta'][t, :]
    #         print(theta.shape)
    #         # theta[0,:3] = torch.tensor([0,0,0]).reshape(1,-1)
    #         verts, _, _ = smpl_male(beta, theta, get_skin=True)
    #         print(sample['tran'][t, 0, :])
    #         # verts = (verts[0] + sample['tran'][t, 0, :]).numpy()
    #         verts = (verts[0] + sample['tran'][t, 0, :]).numpy()
    #         verts = torch.from_numpy(verts)
    #         # verts = verts.unsqueeze(0)
    #         # faces = smpl_male.faces
    #         # dist = np.abs(np.mean(verts, axis=0)[2])

    #         verts2d = projection(verts, intri, True)
    #         verts2d_pix = verts2d.astype(np.uint16)
    #         mask = np.zeros((256, 256), dtype=np.uint8)
    #         img = curr_fullpic.copy()
    #         img[verts2d_pix[verts2d_pix[:,1]<256,1], verts2d_pix[verts2d_pix[:,0]<256,0]] = [255,255,255]
    #         # render_img = (util.render_model(verts, faces, 256, 256, intri, np.zeros([3]),
    #         #               np.zeros([3]), near=0.1, far=20 + dist, img=curr_fullpic) * 255).astype(np.uint8)
            
    #         plt.subplot(4, 4, 11)
    #         plt.imshow(img)
    #         plt.axis('off')

    #         # joint2d and 3d
    #         img = curr_fullpic.copy()
    #         proj_joints2d = projection(joints3d[t], intri, True)
    #         for point in proj_joints2d.astype(np.int64):
    #             cv2.circle(img, (point[0], point[1]), 1, (0, 0, 255), -1)
    #         plt.subplot(4, 4, 12)
    #         plt.imshow(img)
    #         plt.axis('off')
    #         plt.title('joints3d')

    #         img = curr_fullpic.copy()
    #         for point in joints2d[t].astype(np.int64):
    #             cv2.circle(img, (point[0], point[1]), 1, (255, 0, 0), -1)
    #         plt.subplot(4, 4, 13)
    #         plt.imshow(img)
    #         plt.axis('off')
    #         plt.title('joints2d')

    #         flow_rgb = flow_to_image(flows[t])
    #         plt.subplot(4, 4, 14)
    #         plt.imshow(flow_rgb)
    #         plt.axis('off')
    #         plt.title('flow')

    #         plt.show()

    #     # '''

def findBoundaryPixels(vertPixels, H=256):
    boundaryPixels = []
    vertPixels = vertPixels.astype(np.int8)
    for iImg in range(vertPixels.shape[0]):
        mask = np.zeros((H, H), dtype=np.uint8)
        mask[vertPixels[iImg,:,1], vertPixels[iImg,:,0]] = 255
        # Dilate the mask to create circles
        kernel7x7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask7x7 = cv2.dilate(mask, kernel7x7)
        kernel5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask5x5 = cv2.dilate(mask, kernel5x5)
        image7x7 = np.full((H, H), 0, dtype=np.uint8)  # Grey background
        image5x5 = np.full((H, H), 0, dtype=np.uint8)  # Grey background
        # Apply the mask to the image
        # image7x7[mask7x7 == 255] += np.array([0, 127, 127]).astype(np.uint8)  # White dots
        # image5x5[mask5x5 == 255] += np.array([127, 127, 127]).astype(np.uint8)  # White dots
        image7x7[mask7x7 == 255] += np.array([255]).astype(np.uint8)
        image5x5[mask5x5 == 255] += np.array([255]).astype(np.uint8)
        edge = image7x7 - image5x5
        non_zero_locations = np.nonzero(edge != 0)
        non_zeros = np.array(list(zip(non_zero_locations[0], non_zero_locations[1])))
        boundaryPixels.append(non_zeros)
    return boundaryPixels

def projection(xyz, intr_param, simple_mode=False):
    # xyz: [N, 3]
    # intr_param: (fx, fy, cx, cy, w, h, k1, k2, p1, p2, k3, k4, k5, k6)
    assert xyz.shape[1] == 3
    fx, fy, cx, cy = intr_param[0:4]

    if not simple_mode:
        k1, k2, p1, p2, k3, k4, k5, k6 = intr_param[6:14]

        x_p = xyz[:, 0] / xyz[:, 2]
        y_p = xyz[:, 1] / xyz[:, 2]
        r2 = x_p ** 2 + y_p ** 2

        a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        b = b + (b == 0)
        d = a / b

        x_pp = x_p * d + 2 * p1 * x_p * y_p + p2 * (r2 + 2 * x_p ** 2)
        y_pp = y_p * d + p1 * (r2 + 2 * y_p ** 2) + 2 * p2 * x_p * y_p

        u = fx * x_pp + cx
        v = fy * y_pp + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)
    else:
        u = xyz[:, 0] / xyz[:, 2] * fx + cx
        v = xyz[:, 1] / xyz[:, 2] * fy + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    data_train = TrackingDataloader(
        max_steps=8,
        num_steps=8,
        skip=2,
        events_input_channel=8,
        img_size=256,
        mode='all',
        use_hmr_feats=False
    )
    # sample = data_train[10000]
    data_train.visualize(14000)
    # print()
    # for k, v in sample.items():
    #     if k is not 'info':
    #         print(k, v.size())

    # data_test = TrackingDataloader(
    #     max_steps=16,
    #     num_steps=8,
    #     skip=2,
    #     events_input_channel=8,
    #     img_size=256,
    #     mode='train',
    #     use_flow=True,
    #     use_flow_rgb=False,
    #     use_hmr_feats=False,
    #     use_vibe_init=False,
    #     use_hmr_init=True,
    # )
    # # data_test.visualize(30000)
    # sample = data_test[30000]
    # for k, v in sample.items():
    #     if k != 'info':
    #         print(k, v.size())

