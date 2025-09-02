# Whole Slide Image (WSI) Segmentation Pipeline

This project provides a complete pipeline to perform **whole slide image (WSI) segmentation** using a patch-based approach. The pipeline consists of patch generation, vessel and lobule segmentation using separate U-Net models, blending their outputs, overlaying on original image patches, and finally stitching everything together into a complete segmentation output.

## 📌 Pipeline Overview

### 1. **Patch Generator**
- **Script**: `Patch Generator.py`
- **Function**: 
  - Accepts `.ndpi` WSI images.
  - Generates smaller image patches from the WSI.
  - These patches are input to segmentation models in later steps.

### 2. **Vessel Segmentation**
- **Script**: `GeneratorV.py`
- **Function**: 
  - Takes patches as input.
  - Applies a U-Net model trained for **vessel segmentation**.
  - Outputs vessel segmentation masks for each patch.

### 3. **Lobule Segmentation**
- **Script**: `Generator L.py`
- **Function**: 
  - Takes the same patches.
  - Applies a U-Net model trained for **lobule segmentation**.
  - Outputs lobule segmentation masks.

### 4. **Prediction Blending**
- **Script**: `ImageBlending.py`
- **Function**: 
  - Blends vessel and lobule segmentation outputs.
  - Applies logic from code to ensure effective combination of masks.

### 5. **Overlay Creation**
- **Script**: `ImageOverlay.py`
- **Function**: 
  - Overlays blended prediction masks on top of the original image patches.
  - Helps in visual inspection of segmentation quality.

### 6. **Image Stitching**
- **Script**: `Image Stitching.py`
- **Function**: 
  - Reconstructs the full WSI by stitching all the overlayed patches.
  - Outputs a complete segmentation map for the entire slide.


## 💻 Software Environment

| Dependency      | Version    |
|----------------|------------|
| Python          | 3.10       |
| PyTorch         | 2.6        |
| CUDA            | 11.8       |

