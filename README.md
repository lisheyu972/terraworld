# OminiControl Training and Inference Bundle for Four Tasks

This bundle contains training and inference launchers for four tasks: fMOW frame prediction, NIR-to-optical translation, SAR-to-optical translation, and fMOW inpainting. The `translation/` directory was copied from `186_backups/OminiControl-main`; the `inpaint/` directory was copied from `grsm/OminiControl-main-183`. Each keeps its own version of the Python modules so the inpainting code is not mixed with the translation code.

The bundle includes the eight requested launchers, their configuration templates and inference tools, and the local Python modules they import: `omini/`, `glor_control/`, `toa_control/`, `scb_control/`, `observation_residual/`, and `metadata_qkv.py`. The SAR launcher also includes the helper launcher that it calls. Large datasets, the base FLUX model, pretrained Transformers, and trained LoRA checkpoints remain external and are not copied into this directory.

## Install the environment

Use Linux with an NVIDIA GPU and a CUDA-compatible PyTorch installation. For example:

```bash
conda create -n omini-four-tasks python=3.10 -y
conda activate omini-four-tasks
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r /data/dxc/omini_four_tasks_bundle/requirements.txt
```

If PyTorch cannot detect CUDA, install a GPU build of PyTorch that matches your driver and CUDA environment, then install `requirements.txt`. Copies of the original training dependency lists are retained as `requirements-source.txt` in each task directory. The bundle-level requirements file covers the dependencies used by these four tasks.

## Set external paths

```bash
cd /data/dxc/omini_four_tasks_bundle
cp paths.env.example paths.env
# Edit paths.env and replace any missing locations with absolute paths on your machine.
```

`paths.env.example` contains paths used on the original machine. You can also set these variables in the environment before invoking a launcher.

| Variable | Purpose |
| --- | --- |
| `FLUX_MODEL_DIR` | Complete FLUX.1-dev Diffusers pipeline, including the VAE, text encoders, and tokenizers |
| `FRAME_TRANSFORMER_DIR` | Pretrained Transformer for fMOW frame prediction |
| `SAR_TRANSFORMER_DIR` | Pretrained Transformer for SAR-to-optical translation |
| `INPAINT_TRANSFORMER_DIR` | Pretrained Transformer for fMOW inpainting |
| `FMOW_TRAIN_ROOT`, `FMOW_TRAIN_INDEX` | fMOW training data and its `index.jsonl` |
| `FMOW_TEST_ROOT`, `FMOW_TEST_INDEX` | fMOW inpainting test data and its `index.jsonl` |
| `NIR_DATA_ROOT`, `SAR_DATA_ROOT` | Paired NIR and SAR datasets |

Each launcher locates its own directory and generates an absolute-path YAML configuration under its task directory's `runtime/config/`. New training runs are written to that task directory's `runs/`; inference results are written to `outputs/`. If you move the bundle, the launchers recalculate their internal paths from the new location. Relative values passed through `CKPT`, `CONFIG`, or `OUTPUT_DIR` are resolved from the corresponding task directory.

## Train

These launchers can be invoked from any working directory. They retain the original defaults for GPUs, batch size, and training steps. Override `GPU_IDS`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `MAX_STEPS`, or `SAVE_EVERY_N_STEPS` as needed.

```bash
bash /data/dxc/omini_four_tasks_bundle/translation/train/script/train_fmow_frame_prediction_more_layers_glor_control_miq_mma.sh
bash /data/dxc/omini_four_tasks_bundle/translation/train/script/train_nir_to_optical_more_layers_glor_control.sh
bash /data/dxc/omini_four_tasks_bundle/translation/train/script/train_sar_to_optical_more_layers_glor_control.sh
bash /data/dxc/omini_four_tasks_bundle/inpaint/train/script/train_fmow_inpaint_more_layers_glor_control_miq_mma.sh
```

For example, to train fMOW frame prediction on four GPUs:

```bash
GPU_IDS=0,1,2,3 BATCH_SIZE=2 MAX_STEPS=15000 \
  bash /data/dxc/omini_four_tasks_bundle/translation/train/script/train_fmow_frame_prediction_more_layers_glor_control_miq_mma.sh
```

`BATCH_SIZE` is the per-GPU batch size. The effective global batch size is `BATCH_SIZE × GRAD_ACCUM_STEPS × number of GPUs`. These launchers initialize new LoRA weights from the pretrained model. Resuming the complete training state of an earlier run requires a separate implementation.

## Run inference

The default `CKPT` values point to historical checkpoints in the original projects. Set `CKPT` explicitly when moving the bundle or using a newly trained checkpoint. A checkpoint must contain `text.safetensors`, `main.safetensors`, `condition.safetensors`, and `glor_control.pt`. The two MIQ-MMA tasks also require `metadata_qkv.pt`. By default, `GLOR_CKPT` resolves to `CKPT/glor_control.pt`.

```bash
CKPT=/absolute/path/to/frame/ckpt/15000 GPU_IDS=0,1,2,3 \
  bash /data/dxc/omini_four_tasks_bundle/translation/train/script/infer_fmow_frame_prediction_more_layers_glor_control_miq_mma_rivg_stable_strong.sh

CKPT=/absolute/path/to/nir/ckpt/15000 GPU_IDS=0 \
  bash /data/dxc/omini_four_tasks_bundle/translation/train/script/infer_nir_to_optical_more_layers_glor_control_rivg_stable.sh

CKPT=/absolute/path/to/sar/ckpt/15000 PRESET=16000 GPU_IDS=0 \
  bash /data/dxc/omini_four_tasks_bundle/translation/train/script/infer_sar_to_optical_more_layers_glor_control_rivg_stable.sh

CKPT=/absolute/path/to/inpaint/ckpt/15000 GPU_IDS=0,1,2,3 \
  bash /data/dxc/omini_four_tasks_bundle/inpaint/train/script/infer_fmow_inpaint_more_layers_glor_control_miq_mma_rivg_stable_strong.sh
```

The SAR launcher retains `PRESET=30000|16000|both`; its default, `both`, runs the two presets in sequence. Select a single preset when supplying a custom checkpoint. Additional inference arguments can be appended to a launcher command, such as `--max-samples 100`. Inference uses unshifted inputs and retains each launcher's original RIVG settings.

## Path and file notes

- Launchers resolve their code entry points, generated configurations, run directories, and output directories to absolute paths. They do not depend on the shell's starting directory.
- `train/config/*.yaml` files are templates. Runs use the generated `runtime/config/*.yaml` files, whose model, dataset, and output paths are absolute.
- Original `runs/` and `outputs/` directories, datasets, and large model weights were not copied. Update `paths.env` and the inference `CKPT` value after moving the bundle.
- This was a code and path cleanup only. The environment was not installed, and training, inference, and tests were not run.
