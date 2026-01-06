# Mammogram Detector – Deep Learning Breast Tissue Classifier

A Python-based deep learning project that predicts whether mammogram images show normal or abnormal breast tissue, potentially indicating breast cancer.  

## Overview

This project processes a dataset of mammogram images and trains a **Convolutional Neural Network (CNN)** to classify breast tissue as **Normal** or **Abnormal**.  

Key highlights of the project:
- Preprocessing of medical imaging data
- Training a CNN with **adaptive handling of class imbalance**
- Evaluation on unseen test data
- Visualization of results including confusion matrices, classification reports, and sample predictions
- Optional GUI for predicting images from a folder

## Key Features

- Image preprocessing, resizing, and normalization
- Data augmentation using `ImageDataGenerator`
- CNN architecture with convolution, pooling, dropout, and dense layers
- Adaptive class weight adjustment based on confusion matrix after each epoch
- Training history saving and visualization
- Evaluation metrics: accuracy, classification report, confusion matrix
- Tkinter-based interface for folder-based prediction and visualization

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, OpenCV
- Matplotlib
- scikit-learn
- Tkinter (GUI)

## Project Status

- 🟢 Functional
- Tested on held-out dataset
- GUI for predictions implemented

## How It Works

1. **Preprocessing:**  
   - Load mammogram images and associated labels  
   - Resize images to `(224, 224)` and normalize pixel values  
   - Split into train and test sets  
   - Apply data augmentation

2. **Model Training:**  
   - Build CNN architecture with multiple convolution and pooling layers  
   - Train using augmented data  
   - Use **adaptive class weighting** based on confusion matrix at each epoch to address class imbalance  
   - Save the trained model and training history

3. **Evaluation & Visualization:**  
   - Test on unseen data  
   - Generate accuracy metrics, classification report, and confusion matrix  
   - Save sample prediction images for visual verification

4. **GUI Prediction (Optional):**  
   - Load trained model  
   - Select a folder of images  
   - Display predictions for each image with probability scores

## Running Locally

### Prerequisites

- Python 3.x  
- TensorFlow / Keras  
- OpenCV, NumPy, Pandas, scikit-learn, Matplotlib, PIL (Pillow)  

### Setup

```bash
git clone <repo-url>
cd mammogram-detector
pip install -r requirements.txt
python main.py

Ensure dataset files (.h5 and labels) are in the expected paths.

GUI will launch for folder-based predictions.

What This Repo Demonstrates

Building and training a CNN for medical image classification
Handling class imbalance in a creative, adaptive way
Preprocessing and augmentation of real-world imaging datasets
Visualization of model performance and predictions
Integration of ML model with a simple user interface
