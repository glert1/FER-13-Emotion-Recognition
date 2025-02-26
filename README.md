# FER-13 Emotion Recognition

This repository contains an implementation of facial emotion recognition using deep learning models trained on the FER-13 dataset.

## Project Overview

The project consists of three main components:

1. **Data Loading** (`dataloader.py`):
   - Loads and preprocesses the FER-13 dataset.
   - Applies transformations such as resizing, normalization, and augmentation.
   - Creates PyTorch dataloaders for training and testing.

2. **Model Training** (`train.py`):
   - Supports multiple deep learning models (VGG16, ResNet50, InceptionV3).
   - Uses transfer learning to fine-tune pre-trained models.
   - Implements a training loop with cross-entropy loss and Adam optimizer.
   - Saves the trained model checkpoint.

3. **Model Evaluation** (`test.py`):
   - Loads the trained model and evaluates it on the test dataset.
   - Computes accuracy and displays a confusion matrix.
   - Generates a classification report for detailed performance analysis.

## Requirements

Before running the scripts, install the necessary dependencies:

```bash
pip install torch torchvision tqdm numpy matplotlib scikit-learn pillow
```

## Dataset

The project uses the **FER-13 (Facial Expression Recognition 2013) dataset**, which consists of grayscale images categorized into 7 emotions:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

Make sure the dataset is structured in the following way:
```
dataset/
│── train/
│   ├── Angry/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Neutral/
│   ├── Sad/
│   ├── Surprise/
│── test/
    ├── Angry/
    ├── Disgust/
    ├── Fear/
    ├── Happy/
    ├── Neutral/
    ├── Sad/
    ├── Surprise/
```

## Usage

### Train the Model
```bash
python train.py
```
Modify `train.py` to change the model architecture or training parameters.

### Evaluate the Model
```bash
python test.py
```
Ensure that the trained model (`vgg16_trained.pth`, `resnet50_trained.pth`, etc.) is available before running the evaluation.

## Output
- The model's accuracy will be displayed after training.
- A confusion matrix and classification report will be generated after evaluation.

## License
This project is licensed under the MIT License.

