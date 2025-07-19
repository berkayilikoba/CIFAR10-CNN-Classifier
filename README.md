# CIFAR-10 CNN Classifier

A simple deep learning model using a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset.

## Project Overview

This project implements a CNN to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes. The model is built using Keras and TensorFlow.

## Features

- Loads and preprocesses the CIFAR-10 dataset
- Defines a CNN architecture suitable for image classification
- Trains the model with training data
- Evaluates the model on the test data
- Saves the trained model for future use

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Seaborn and Matplotlib (optional, for visualization)

## Installation

You can install the required packages via pip:

```bash
pip install tensorflow keras numpy matplotlib seaborn
```

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. It is widely used for training machine learning and computer vision algorithms. The dataset is divided into 50,000 training images and 10,000 test images.

## Usage

model.py — Defines the CNN model architecture.
train.py — Loads data, trains the model, and saves the best model.
predict.py — Loads the saved model, performs predictions on test data, and visualizes results.
