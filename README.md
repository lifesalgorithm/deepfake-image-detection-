Deepfake Image Detection
This project is a deep learning-based pipeline to detect deepfake images using transfer learning.
It compares the performance of ResNet50 and EfficientNet-B0 on a custom dataset of real and fake images.

Project Overview
Goal: Classify images as real or fake (deepfake).
Dataset: Deepfake image dataset downloaded from Kaggle.
Models Used:
ResNet50: A residual neural network pre-trained on ImageNet.
EfficientNet-B0: A state-of-the-art efficient architecture for image classification.
Frameworks: PyTorch, torchvision, EfficientNet-PyTorch, OpenCV, and scikit-learn.

Key Features
Data Preprocessing:
Image loading, validation, resizing (224x224).
Data augmentation with random flips and rotations.
Class balancing using RandomOverSampler.

Model Training:
Transfer learning with frozen base weights.
Custom fully connected layers for binary classification.
Adaptive learning rate scheduler.
Early stopping based on validation loss.

Evaluation & Visualization:
Test accuracy calculation.
Confusion matrix for performance insight.
Visualization of predictions vs. actual labels.
Comparison of ResNet50 and EfficientNet performance.

Results
ResNet50 Test Accuracy: ~81.63%
EfficientNet-B0 Test Accuracy: ~86.31%

EfficientNet-B0 performed better on this dataset.


How it Works
Data Loading & EDA: Checks for corrupt images, missing data, class imbalance, and visualizes samples.
Preprocessing: Resizes images, augments training data.
Custom Dataset: PyTorch Dataset class handles transformations and labels.
Model Definition: Loads pre-trained ResNet50 and EfficientNet-B0; modifies final layers for binary output.
Training Loop: Uses BCE loss, Adam optimizer, and scheduler.
Evaluation: Tests on hold-out data, visualizes predictions, plots confusion matrix.
Model Selection: Saves the best-performing model.

Next Steps / Improvements
Use more advanced augmentation techniques.
Test with other EfficientNet versions or Vision Transformers.
Try video-based detection for deepfake videos.

Deploy the model as an API for real-time detection.

