# DiWINE-VLA: DIstributed World-Intrinsic Neural Embedding for VLA

DiWINE-VLA extends X-VLA with world-intrinsic latent prediction for vision-language-action learning. The current codebase builds on a Florence2 vision-language encoder, a domain-aware soft-prompted Transformer action head, and optional future latent modules for learning predictive structure from robot demonstrations.

This repository is under active development. Results and benchmark tables will be added as experiments complete.

## Overview

DiWINE-VLA is designed for cross-embodiment robot policy learning from heterogeneous demonstrations. The model takes multi-view images, a language instruction, proprioception, and an embodiment/domain id, then predicts a horizon of robot actions through a flow-matching style denoising process.

Core components:

- **Vision-language encoder**: Florence2 encoder-only backbone for language-conditioned visual features.
- **Soft-prompted action Transformer**: domain-conditioned soft prompts and domain-aware action encoders/decoders.
- **World-intrinsic latent module**: optional future-token prediction using compressed latent visual states, proprioceptive conditioning, and RSSM-style rollout.
- **Unified action spaces**: support for EE6D, joint, AGIBOT EE6D, and auto-padded action formats.
- **Server-client deployment**: FastAPI inference service for simulator or robot clients.

## Repository Layout

```text
models/
  modeling_xvla.py          # Top-level VLA model and inference server
  transformer.py            # Soft-prompted Transformer action head
  action_hub.py             # Action-space registry, losses, and postprocessing
  processing_xvla.py        # Multi-view image and language processor
  configuration_xvla.py     # Model configuration

datasets/
  dataset.py                # Meta-file based iterable dataset
  rlds_dataset.py           # Native DROID RLDS reader
  domain_handler/           # Dataset-specific trajectory loaders
  domain_config.py          # Dataset weights and domain ids

train.py                    # Full fine-tuning / training entry point
peft_train.py               # LoRA fine-tuning entry point
deploy.py                   # FastAPI inference server launcher
evaluation/                 # Benchmark and robot client scripts
```

## Installation

```bash
conda create -n diwine-vla python=3.10 -y
conda activate diwine-vla
pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate xvla-stable
```

## Training

Example BF16 mixed-precision training command:

```bash
accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  train.py \
  --models "2toINF/X-VLA-Libero" \
  --train_metas_path /scratch/shahils/openpi/datasets/Libero-XVLA-format/libero_meta.json \
  --batch_size 32 \
  --num_workers 4 \
  --num_views 3 \
  --learning_rate 1e-4 \
  --output_dir /scratch/shahils/X-VLA/xvla_checkpoints
```

Useful options:

| Argument | Description |
| :--- | :--- |
| `--models` | Base checkpoint or local model directory. |
| `--train_metas_path` | Path to one meta JSON file or a directory of meta JSON files. |
| `--batch_size` | Per-process dataloader batch size. |
| `--num_views` | Number of camera views expected by the processor/model. |
| `--learning_rate` | Base learning rate for trainable modules. |
| `--freeze_steps` | Initial steps with selected backbone/core groups frozen. |
| `--use_future_prediction` | Enable future latent prediction modules. |
| `--future_offset_steps` | Offset from current frame to future supervision frame(s). |
| `--future_horizon_steps` | Number of future latent targets to supervise. |
| `--use_future_contrastive` | Add multi-depth contrastive alignment between current and future encodings. |

### Future Latent Training

The future-prediction path can be enabled with:

```bash
--use_future_prediction \
--future_offset_steps 5 \
--future_horizon_steps 1
```

When enabled, the model compresses current Florence2 encoder states into latent future tokens, predicts future latent states with an RSSM-style predictor, and injects sampled future tokens back into the remaining Florence2 encoder layers before action prediction.

## Data Format

Training uses meta JSON files that point to trajectory files. Each dataset is decoded through a registered domain handler.

Expected sample fields after dataset processing:

| Field | Shape / Type | Description |
| :--- | :--- | :--- |
| `language_instruction` | `str` | Natural-language task instruction. |
| `image_input` | `[V, C, H, W]` | Multi-view image tensor. |
| `image_mask` | `[V]` | Boolean mask for valid image views. |
| `domain_id` | scalar tensor | Embodiment/domain id used for soft prompts and domain-aware layers. |
| `proprio` | `[D]` | Current robot state. |
| `action` | `[T, D]` | Future action trajectory. |
| `future_image_input` | optional `[F, V, C, H_img, W_img]` | Future visual targets for latent prediction. |
| `future_proprio` | optional `[F, D]` | Future proprioceptive targets for latent prediction. |

To add a new dataset:

1. Create a meta JSON with `dataset_name`, optional `robot_type`, and `datalist`.
2. Implement a handler under `datasets/domain_handler/`.
3. Register the handler in `datasets/domain_handler/registry.py`.
4. Add the dataset weight and `domain_id` in `datasets/domain_config.py`.

## Inference Server

Launch a model server:

```bash
python -m deploy \
  --model_path /path/to/model_or_checkpoint \
  --processor_path /path/to/processor_or_checkpoint \
  --port 8010
```

The server exposes:

```text
POST http://<server_ip>:8010/act
```

Expected request fields:

| Key | Description |
| :--- | :--- |
| `proprio` | JSON-serialized proprioceptive state via `json_numpy.dumps`. |
| `language_instruction` | Task instruction string. |
| `image0` | Primary RGB image via `json_numpy.dumps`. |
| `image1`, `image2` | Optional additional camera views. |
| `domain_id` | Embodiment/domain id. |
| `steps` | Number of inference denoising steps. |

## Action Spaces

Supported action modes are registered in `models/action_hub.py`:

| Mode | Description |
| :--- | :--- |
| `ee6d` | 20D end-effector action layout for dual-arm-compatible control. |
| `joint` | Joint-space action layout with gripper channels. |
| `agibot_ee6d` | EE6D variant using MSE for all action components. |
| `auto` | Pads/trims arbitrary real action dimensions to a model-facing max dimension. |

For EE6D, the 20D layout is:

```text
arm_1: xyz(3) + rotation_6d(6) + gripper(1)
arm_2: xyz(3) + rotation_6d(6) + gripper(1)
```

Single-arm datasets can pad the unused arm channels with zeros.

## Evaluation

Evaluation clients are organized under `evaluation/`. Current folders include:

- `evaluation/libero/`
- `evaluation/calvin/`
- `evaluation/simpler/`
- `evaluation/vlabench/`
- `evaluation/robotwin-2.0/`
- `evaluation/SoftFold-Agilex/`

## Results

Results will be added as experiments complete.

| Benchmark | Setting | Metric | Result |
| :--- | :--- | :--- | :--- |
| LIBERO | TBD | Success rate | TBD |
| CALVIN | TBD | Average sequence length | TBD |
| RoboTwin2 | TBD | Success rate | TBD |

## Acknowledgements

This work builds on the X-VLA codebase and model interface. Please cite the original X-VLA work when using components derived from that project.

```bibtex
@article{zheng2025x,
  title   = {X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model},
  author  = {Zheng, Jinliang and Li, Jianxiong and Wang, Zhihao and Liu, Dongxiu and Kang, Xirui
             and Feng, Yuchun and Zheng, Yinan and Zou, Jiayin and Chen, Yilun and Zeng, Jia and others},
  journal = {arXiv preprint arXiv:2510.10274},
  year    = {2025}
}
```

## License

This repository currently follows the upstream Apache License 2.0 terms.
