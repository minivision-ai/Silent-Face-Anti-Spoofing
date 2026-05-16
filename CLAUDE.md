# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Silent-Face-Anti-Spoofing is a face liveness detection (anti-spoofing) system by Minivision. It classifies faces as real or fake (printed photos, screens, masks) using a Fourier spectrum auxiliary supervision approach. The project uses PyTorch.

## Commands

### Install dependencies
```
pip install -r requirements.txt
```

### Train
```
python train.py --device_ids 0 --patch_info 1_80x80
```
- `--device_ids`: GPU IDs (e.g., `0`, `01` for multi-GPU)
- `--patch_info`: patch format, one of `org_1_80x60`, `1_80x80`, `2.7_80x80`, `4_80x80`

### Test (inference on a single image)
```
python test.py --image_name image_F1.jpg --model_dir ./resources/anti_spoof_models --device_id 0
```
Test images go in `./images/sample/`. Images must have 3:4 width:height ratio.

## Architecture

### Multi-scale fusion approach
The system uses multiple models at different scales (crop patches) that are fused at inference time. Each model is trained on a specific patch type:
- `org_1_80x60`: full original image resized
- `1_80x80`, `2.7_80x80`, `4_80x80`: face crops at different scales around the detected bounding box

At test time, predictions from all models in `resources/anti_spoof_models/` are summed and argmax'd across 3 classes (label 1 = real).

### Model architecture (`src/model_lib/`)
- **MiniFASNet** (`MiniFASNet.py`): Pruned MobileNet-style backbone with depthwise separable convolutions. Four variants: `MiniFASNetV1`, `V2`, `V1SE`, `V2SE` (SE = Squeeze-and-Excitation). Channel configs are in `keep_dict`.
- **MultiFTNet** (`MultiFTNet.py`): Training wrapper that adds a Fourier Transform auxiliary branch (`FTGenerator`) to `MiniFASNetV2SE`. During training, returns both classification logits and FT feature maps; during inference, returns only classification. The total loss is `0.5 * cross_entropy + 0.5 * MSE(ft_prediction, ft_label)`.

### Key convention: model filename encoding
Model filenames encode architecture params: `{scale}_{height}x{width}_{ModelType}.pth` (e.g., `2.7_80x80_MiniFASNetV2.pth`). Parsed by `src/utility.py:parse_model_name()`. The `conv6_kernel` size is computed from input dimensions: `((h+15)//16, (w+15)//16)`.

### Data pipeline (`src/data_io/`)
- `dataset_folder.py`: `DatasetFolderFT` — loads images from class-labeled subdirectories (0/1/2) and generates Fourier spectrum labels on-the-fly
- `dataset_loader.py`: builds DataLoader with augmentation (random crop, color jitter, rotation, flip)
- `transform.py` / `functional.py`: custom transforms wrapping torchvision

### Face detection
Uses OpenCV DNN with a Caffe RetinaFace model (`resources/detection_model/`) for bounding box detection. Implemented in `src/anti_spoof_predict.py:Detection`.

### Training outputs
- Model snapshots: `saved_logs/snapshot/{job_name}/`
- TensorBoard logs: `saved_logs/jobs/{job_name}/`

### Dataset structure
```
datasets/rgb_image/{patch_info}/
    ├── 0/   (class 0)
    ├── 1/   (class 1 = real face)
    └── 2/   (class 2)
```