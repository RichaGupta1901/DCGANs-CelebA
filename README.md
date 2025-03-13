# Deep Convolutional Generative Adversarial Network (DCGAN) on CelebA Dataset

## Overview
This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic human face images. The model is trained on the CelebA dataset using PyTorch in Google Colab.

## Requirements
Ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

## Dataset Preprocessing
1. The CelebA dataset is downloaded and loaded.
2. Images are resized to `64x64` pixels.
3. Images are normalized to the range `[-1, 1]`.

## Model Architecture
### Generator
- Uses `ConvTranspose2d` layers to upscale the noise vector into a `64x64` image.
- Applies `BatchNorm` and `ReLU` activation functions.
- Outputs a `tanh`-activated image with three channels (RGB).

### Discriminator
- Uses `Conv2d` layers to downsample images.
- Applies `LeakyReLU` activation for better gradient flow.
- Outputs a single probability value through a `sigmoid` activation.

## Training
1. **Discriminator Training:**
   - Trained on both real and fake images.
   - Uses **Binary Cross Entropy Loss (BCELoss)**.
   - Optimized using **Adam optimizer**.

2. **Generator Training:**
   - Generates fake images from random noise.
   - Tries to fool the discriminator.
   - Uses BCELoss and Adam optimizer.

## Expected Output
- Initially, generated images will be noisy and unrecognizable.
- As training progresses, the quality of generated images improves.
- After `10 epochs`, the model should generate **human-like** face images.

## References
- Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (2015)
