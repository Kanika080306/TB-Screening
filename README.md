# Tuberculosis Detection Using Chest X-Ray Images

A deep learning-based approach for classifying chest X-ray images into normal and abnormal categories using image preprocessing, lung segmentation, and transfer learning.

## Overview

This project explores the application of deep learning techniques for tuberculosis-related chest X-ray classification using the **Montgomery County Chest X-ray Dataset**.

The pipeline incorporates lung segmentation masks to isolate relevant lung regions, image preprocessing to standardize inputs, and transfer learning to extract visual features for binary classification.

It also investigates the impact of different optimizers and learning rates on model performance.

## Features

- Chest X-ray image preprocessing and normalization
- Lung region extraction using segmentation masks
- Image resizing and padding to standard dimensions
- Transfer learning with pretrained feature extraction
- Binary classification of normal and abnormal images
- Comparison of multiple optimizers and learning rates
- Early stopping and class weighting
- Evaluation using multiple classification metrics

## Dataset

**Montgomery County Chest X-ray Dataset**

The dataset contains chest radiographs and corresponding lung segmentation masks.

The classification labels used in the current implementation are:

| Label | Class |
|---|---|
| 0 | Normal |
| 1 | Abnormal |

The dataset's abnormal category is used as the positive class in the classification pipeline. An abnormal radiograph should not automatically be interpreted as a confirmed TB diagnosis.

## Methodology

The project follows the workflow below:

1. **Data Loading:** Load chest X-ray images and corresponding lung masks.
2. **Image Preprocessing:** Convert images to grayscale and normalize pixel values.
3. **Lung Segmentation:** Apply binary masks to retain the lung regions.
4. **Image Standardization:** Resize images to 224 × 224 while preserving aspect ratio, followed by center padding.
5. **Dataset Splitting:** Divide the data into training, validation, and testing subsets using stratified sampling.
6. **Feature Extraction:** Extract visual features using a pretrained convolutional neural network.
7. **Classification:** Train a binary classification head to distinguish between the two classes.
8. **Hyperparameter Experiments:** Compare different optimizers and learning rates.
9. **Model Selection:** Select the best configuration based on validation loss.
10. **Evaluation:** Evaluate the selected model on the held-out test set.

## Model Architecture

The current implementation uses transfer learning with a pretrained convolutional neural network.

The architecture consists of:

- Pretrained feature extraction backbone with ImageNet weights
- Frozen pretrained layers
- Global Average Pooling
- Dropout layer (0.4)
- Fully connected output layer
- Sigmoid activation for binary classification

**Transfer Learning Strategy:** Feature extraction. The pretrained backbone remains frozen while the newly added classification head is trained on the target dataset.

## Training Configuration

| Parameter | Value |
|---|---|
| Input Image Size | 224 × 224 |
| Input Channels | 3 |
| Batch Size | 8 |
| Maximum Epochs | 20 |
| Early Stopping Patience | 3 |
| Dropout Rate | 0.4 |
| Loss Function | Binary Cross-Entropy |
| Output Activation | Sigmoid |
| Class Weighting | Balanced |
| Dataset Split | 70% Training / 10% Validation / 20% Testing |

### Optimizer Comparison

The training pipeline evaluates three optimizers across two learning rates.

| Optimizer | Learning Rates |
|---|---|
| Adam | 0.0001, 0.001 |
| SGD | 0.0001, 0.001 |
| RMSprop | 0.0001, 0.001 |

A total of six experimental configurations are compared.

Early stopping monitors validation loss and restores the weights from the best-performing epoch.

## Evaluation Metrics

The selected model is evaluated using:

- **Accuracy:** Overall classification correctness.
- **Precision:** Reliability of positive predictions.
- **Recall (Sensitivity):** Ability to identify actual positive samples.
- **F1-Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Distribution of correct and incorrect predictions.
- **ROC-AUC:** Ability to distinguish between the two classes across classification thresholds.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd TB-Detection
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- Python
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Pillow
- Scikit-learn

## Usage

1. Download and organize the Montgomery County dataset.
2. Place the chest X-ray images in the `img` directory.
3. Place the corresponding segmentation masks in the `mask` directory.
4. Update the dataset and output directory paths in `validation.py`.
5. Run the training script:

```bash
python validation.py
```

The script performs preprocessing, dataset splitting, optimizer experiments, model selection, and final evaluation.

## Outputs

The pipeline generates the following outputs:

| Output | Description |
|---|---|
| `densenet121_tb_best_experiment.keras` | Best selected experimental model |
| `optimizer_comparison.csv` | Comparison of optimizer configurations |
| `tb_optimizer_experiments/` | Individual experiment checkpoints |

Evaluation metrics are displayed after the final testing stage.

## Future Work

Potential extensions to the project include:

- Fine-tuning selected pretrained layers
- Experimenting with additional architectures
- Applying data augmentation techniques
- Exploring further hyperparameter optimization
- Integrating explainability techniques such as Grad-CAM
- Evaluating performance on larger, independent datasets
- Improving model generalization and robustness

## Limitations

- The dataset is relatively small, which may affect generalization.
- Abnormal chest X-rays do not necessarily represent confirmed TB cases.
- Model performance may vary depending on the data split.
- The current implementation uses frozen feature extraction rather than fine-tuning.
- Independent validation on larger and more diverse datasets is necessary.

## Disclaimer

This project is developed for academic and research purposes. It is an experimental image classification system and is not intended to replace professional medical evaluation or serve as a clinically validated diagnostic tool.

## Project Status

**Status:** Research and experimental development.
