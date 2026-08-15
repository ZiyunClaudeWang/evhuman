# EvHuman: Continuous-Time Human Motion Field from Event Cameras

### [Project Page](https://ziyunclaudewang.github.io/evhuman/) | [Paper (CVF)](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Continuous-Time_Human_Motion_Field_from_Event_Cameras_ICCV_2025_paper.pdf) | [arXiv](https://arxiv.org/abs/2412.01747)

> **Continuous-Time Human Motion Field from Event Cameras**
> [Ziyun Wang](https://ziyunclaudewang.github.io/)<sup>1,2</sup>, Ruijun Zhang<sup>1</sup>, Zi-Yan Liu<sup>1</sup>, Yufu Wang<sup>1</sup>, [Kostas Daniilidis](https://www.cis.upenn.edu/~kostas/)<sup>1,3</sup>
> <sup>1</sup>University of Pennsylvania, <sup>2</sup>Johns Hopkins University, <sup>3</sup>Archimedes, Athena RC
> **ICCV 2025**

<p align="center">
  <img src="https://ziyunclaudewang.github.io/evhuman/images/qual.png" width="100%">
  <br>
  <em>Qualitative human mesh recovery results from event streams.</em>
</p>

## Abstract

This paper addresses the challenges of estimating a continuous-time human motion field from a stream of events. Existing Human Mesh Recovery (HMR) methods rely predominantly on frame-based approaches, which are prone to aliasing and inaccuracies due to limited temporal resolution and motion blur. In this work, we predict a continuous-time human motion field directly from events by leveraging a recurrent feed-forward neural network to predict human motion in the latent space of possible human motions. Prior state-of-the-art event-based methods rely on computationally intensive optimization across a fixed number of poses at high frame rates, which becomes prohibitively expensive as we increase the temporal resolution. In comparison, we present the first work that replaces traditional discrete-time predictions with a continuous human motion field represented as a time-implicit function, enabling parallel pose queries at arbitrary temporal resolutions. Despite the promises of event cameras, few benchmarks have tested the limit of high-speed human motion estimation. We introduce Beam-splitter Event Agile Human Motion Dataset — a hardware-synchronized high-speed human dataset to fill this gap. On this new data, our method improves joint errors by 23.8% compared to previous event human methods while reducing the computational time by 69%.

## Key Features

- First feed-forward event-based continuous-time human motion field
- 23.8% improvement in joint errors over prior event-based methods
- 69% reduction in computational time at high frame rates
- Novel event-based contrast maximization loss using vertex optical flow
- New BEAHM dataset with 120 FPS ground-truth human meshes

## Results

### MMHPSD

<div align="center">

| Metric | Result |
|:---:|:---:|
| MPJPE | 65.89 |
| PA-MPJPE | 39.03 |
| PEL-MPJPE | 49.95 |
| PCKh@0.5 | 0.862 |

</div>

### BEAHM

<div align="center">

| Metric | Result |
|:---:|:---:|
| MPJPE | 48.32 |
| PA-MPJPE | 30.08 |
| PEL-MPJPE | 41.08 |
| PCKh@0.5 | 0.915 |

</div>

## Installation

```bash
conda create -n evhuman python=3.10
conda activate evhuman
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

For `nvdiffrast`, follow the [official installation guide](https://nvlabs.github.io/nvdiffrast/#installation).

## Data Setup

### MMHPSD Dataset

Download the MMHPSD dataset from [EventHPE](https://github.com/JimmyZou/EventHPE) and place it under `data/mmhpsd/`. The event data should be placed under `data/mmhpsd_events/`.

### BEAHM Dataset

Our Beam-splitter Event Agile Human Motion Dataset (BEAHM) is available at the [project page](https://ziyunclaudewang.github.io/evhuman.html). Place it under `data/beahm/` with events under `data/beahm_events/`.

### Pre-processing

Convert raw events into event volume images:

```bash
# For MMHPSD
python script/convert_volumes.py --event_folder data/mmhpsd_events --save_folder data/mmhpsd_volumes

# For BEAHM
python script/convert_volumes_ours.py --event_folder data/beahm_events --save_folder data/beahm_volumes
```

### SMPL Model

Download the SMPL model from [SMPL](https://smpl.is.tue.mpg.de/) and place:
- `basicModel_m_lbs_10_207_0_v1.0.0.pkl` in `event_hpe/smpl_model/`
- `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` in `event_hpe/smpl_model/`

### NeMF Pre-trained Decoder

The NeMF decoder is pre-trained on the [AMASS](https://amass.is.tue.mpg.de/) dataset. Place the pre-trained decoder weights under `NeMF/outputs/generative/results/model/`.

## Training

All training commands require `PYTHONPATH=NeMF/src:$PYTHONPATH` prefix.

Training follows four stages:

### Stage 1: Train GMP on GT Poses

Train the Global Motion Predictor using ground-truth poses to predict translation:

```bash
PYTHONPATH=NeMF/src:$PYTHONPATH python train_gmp.py \
    --data_root data/mmhpsd \
    --event_folder data/mmhpsd_events/ \
    --result_dir outputs \
    --log_dir gmp_stage1 \
    --hmr_model --ours_full_pose0 --left_mult \
    --pred_traj \
    --use_hmr_feats 0 \
    --batch_size 8 --epochs 10
```

### Stage 2: Train Event Human Motion Predictor

Freeze GMP and train the event encoder + NeMF latent code predictor:

```bash
PYTHONPATH=NeMF/src:$PYTHONPATH python train_eventhpe.py \
    --data_root data/mmhpsd \
    --event_folder data/mmhpsd_events/ \
    --result_dir outputs \
    --log_dir stage2 \
    --hmr_model --ours_full_pose0 --left_mult \
    --pred_traj \
    --use_hmr_feats 0 \
    --batch_size 8 --epochs 10 \
    --gmp_model_path outputs/gmp_stage1/log/<timestamp>/model_events_pose.pkl
```

### Stage 3: Fine-tune GMP on Predicted Poses

Freeze the encoder and fine-tune only the GMP using predicted (not GT) poses:

```bash
PYTHONPATH=NeMF/src:$PYTHONPATH python train_eventhpe.py \
    --data_root data/mmhpsd \
    --event_folder data/mmhpsd_events/ \
    --result_dir outputs \
    --log_dir gmp_finetune \
    --model_dir outputs/stage2/log/<timestamp>/model_events_pose.pkl \
    --hmr_model --ours_full_pose0 --left_mult \
    --pred_traj --only_train_gmp --finetune_gmp --reset_optimizer \
    --use_hmr_feats 0 \
    --batch_size 16 --epochs 1 --lr_start 0.0002
```

### Stage 4: Joint Fine-tuning

Fine-tune all components together for best results:

```bash
PYTHONPATH=NeMF/src:$PYTHONPATH python train_eventhpe.py \
    --data_root data/mmhpsd \
    --event_folder data/mmhpsd_events/ \
    --result_dir outputs \
    --log_dir joint_finetune \
    --model_dir outputs/gmp_finetune/log/<timestamp>/model_events_pose.pkl \
    --hmr_model --ours_full_pose0 --left_mult \
    --pred_traj --finetune_gmp --reset_optimizer \
    --use_hmr_feats 0 \
    --batch_size 32 --epochs 1 --lr_start 0.0001
```

### Training on BEAHM

For BEAHM, add `--our_data`, set `--skip 8`, and disable flow loss. Follow the same 4-stage pipeline:

```bash
# Stage 2: Event encoder (BEAHM)
PYTHONPATH=NeMF/src:$PYTHONPATH python train_eventhpe.py \
    --data_root data/beahm \
    --event_folder data/beahm_events/ \
    --result_dir outputs --log_dir beahm_stage2 \
    --hmr_model --ours_full_pose0 --left_mult \
    --pred_traj --our_data \
    --use_hmr_feats 0 --flow_loss 0 --skip 8 \
    --batch_size 8 --epochs 10

# Stage 3: GMP fine-tune (BEAHM)
PYTHONPATH=NeMF/src:$PYTHONPATH python train_eventhpe.py \
    --data_root data/beahm \
    --event_folder data/beahm_events/ \
    --result_dir outputs --log_dir beahm_gmp \
    --model_dir outputs/beahm_stage2/log/<timestamp>/model_events_pose.pkl \
    --hmr_model --ours_full_pose0 --left_mult \
    --pred_traj --only_train_gmp --finetune_gmp --reset_optimizer --our_data \
    --use_hmr_feats 0 --flow_loss 0 --skip 8 \
    --batch_size 16 --epochs 1 --lr_start 0.0002

# Stage 4: Joint fine-tune (BEAHM)
PYTHONPATH=NeMF/src:$PYTHONPATH python train_eventhpe.py \
    --data_root data/beahm \
    --event_folder data/beahm_events/ \
    --result_dir outputs --log_dir beahm_joint \
    --model_dir outputs/beahm_gmp/log/<timestamp>/model_events_pose.pkl \
    --hmr_model --ours_full_pose0 --left_mult \
    --pred_traj --finetune_gmp --reset_optimizer --our_data \
    --use_hmr_feats 0 --flow_loss 0 --skip 8 \
    --batch_size 16 --epochs 1 --lr_start 0.0001
```

### Contrast Maximization Loss

The event-based contrast maximization loss (Sec. 4.4) provides self-supervised training signal by maximizing the sharpness of the Image of Warped Events (IWE). See [`contrast_maximization/`](contrast_maximization/) for detailed documentation, implementation variants, and per-sequence optimization results.

## Evaluation

### MMHPSD

```bash
PYTHONPATH=NeMF/src:$PYTHONPATH python test_eventhpe.py \
    --data_root data/mmhpsd \
    --event_folder data/mmhpsd_events/ \
    --model_dir ckpts/mmhpsd/model_events_pose.pkl \
    --hmr_model --ours_full_pose0 --left_mult \
    --pred_traj \
    --use_hmr_feats 0
```

### BEAHM at 15 FPS

```bash
PYTHONPATH=NeMF/src:$PYTHONPATH python test_eventhpe.py \
    --data_root data/beahm \
    --event_folder data/beahm_events/ \
    --model_dir ckpts/beahm/model_events_pose.pkl \
    --hmr_model --ours_full_pose0 --left_mult \
    --pred_traj --our_data --skip 8 \
    --use_hmr_feats 0
```

### BEAHM at 120 FPS

```bash
PYTHONPATH=NeMF/src:$PYTHONPATH python test_eventhpe.py \
    --data_root data/beahm \
    --event_folder data/beahm_events/ \
    --model_dir ckpts/beahm/model_events_pose.pkl \
    --hmr_model --ours_full_pose0 --left_mult \
    --pred_traj --our_data --test_high_fps --skip 8 \
    --use_hmr_feats 0
```

## Visualization

### Render Multi-view Video

Generate a smooth high-FPS video with events and multi-view SMPL mesh rendering:

```bash
PYTHONPATH=NeMF/src:$PYTHONPATH python render_video_with_events.py \
    --data_root data/mmhpsd \
    --event_folder data/mmhpsd_events/ \
    --model_dir ckpts/mmhpsd/model_events_pose.pkl \
    --fps 60 --sample_idx 300 --resolution 512
```

The output video shows events (left), front view (center), and side view (right). The continuous-time NeMF decoder is sampled at the target FPS, producing smooth interpolated motion.

### Mesh Overlay Visualization

Generate mesh overlay sequences at variable frame rates:

```bash
PYTHONPATH=NeMF/src:$PYTHONPATH python run_inference.py \
    --data_root data/mmhpsd \
    --event_folder data/mmhpsd_events/ \
    --model_dir ckpts/mmhpsd/model_events_pose.pkl \
    --hmr_model --ours_full_pose0 --left_mult \
    --pred_traj \
    --use_hmr_feats 0 \
    --inference_length 64 \
    --save_folder outputs/viz/
```

## Citation

```bibtex
@inproceedings{wang2025continuous,
    title={{Continuous-Time Human Motion Field from Events}},
    author={Wang, Ziyun and Zhang, Ruijun and Liu, Zi-Yan and Wang, Yufu and Daniilidis, Kostas},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2025}
}
```

## Acknowledgment

This work was supported by NSF FRR 2220868, NSF IIS-RI 2212433, ARO MURI W911NF-20-1-0080, and ONR N00014-22-1-2677. The NeMF decoder is based on [NeMF](https://github.com/c-he/NeMF).
