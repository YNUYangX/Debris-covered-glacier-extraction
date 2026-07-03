# Remote Sensing Glacier Extraction - U-Net + CBAM

A deep learning-based **glacier semantic segmentation** project using **U-Net + CBAM (Convolutional Block Attention Module)** architecture for automatic extraction of glacier areas from multi-spectral remote sensing imagery.

## Overview

This project leverages multi-source remote sensing data, using a U-Net deep learning model with CBAM attention mechanism to achieve high-precision automatic glacier segmentation. It is suitable for glacier change monitoring, climate change research, and related applications.

## Model Architecture

- **Backbone**: U-Net (Encoder-Decoder structure)
- **Attention Mechanism**: CBAM (Channel Attention + Spatial Attention)
- **Loss Function**: Dice Loss + Focal Loss
- **Input**: Multi-band remote sensing imagery (5 bands, 512×512)
- **Output**: Multi-class semantic segmentation map (background, glacier, other)

## Evaluation Metrics

The following metrics are computed for model evaluation:

| Metric | Description |
|--------|-------------|
| Overall Accuracy (OA) | Proportion of correctly classified pixels |
| Precision | Ratio of true positives to total predicted positives |
| Recall | Ratio of true positives to total actual positives |
| F1-Score | Harmonic mean of precision and recall |
| IoU (Jaccard Index) | Intersection over Union per class |
| mIoU | Mean IoU across all classes |
| FWIoU | Frequency-Weighted Intersection over Union |
| Kappa Coefficient | Cohen's Kappa agreement measure |

## File Description

| File | Description |
|------|-------------|
| `Unet CBAM.ipynb` | Main training & prediction notebook: data loading, model definition (U-Net+CBAM), training configuration, prediction, and result mosaicking |
| `Data Augmentation.ipynb` | Training data augmentation for better generalization |
| `Accuracy Evaluation.ipynb` | Accuracy evaluation: confusion matrix, OA, Precision, Recall, F1-Score, IoU, mIoU, FWIoU, Kappa |

## Dependencies

- Python 3.x
- TensorFlow / Keras
- GDAL (osgeo)
- OpenCV (cv2)
- NumPy, Pandas, Matplotlib

## Usage

### 1. Data Preparation

Place remote sensing images and corresponding labels into training, validation, and test directories respectively.

### 2. Model Training

Run the training cells in `Unet CBAM.ipynb`, adjusting hyperparameters such as batch_size, epochs, and learning_rate.

### 3. Prediction

```python
pre_img(TifPath, ResultPath, ModelPath)
```

### 4. Accuracy Evaluation

Run `Accuracy Evaluation.ipynb`, specifying the label path and prediction result path to compute all evaluation metrics.

## References

- U-Net: Convolutional Networks for Biomedical Image Segmentation: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- CBAM: Convolutional Block Attention Module: [Woo et al., 2018](https://arxiv.org/abs/1807.06521)
- Mapping Debris-Covered Glaciers Using High-Resolution Imagery (GF-2) and Deep Learning Algorithms:[Yang et al., 2024](https://doi.org/10.3390/rs16122062)
- High-Resolution Glacier Inventory of the Karakoram:[Yang et al., 2025](https://doi.org/10.12072/ncdc.glacier.db6974.2025)
- High-resolution glacier mapping reveals inventory biases and terrain controls on debris-covered glaciers in the Karakoram:[Yang et al., 2026](https://doi.org/10.5194/egusphere-2026-2836)
