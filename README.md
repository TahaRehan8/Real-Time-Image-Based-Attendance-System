# Real-Time-Image-Based-Attendance-System Using Random Forest
# Random Forest Live Face Recognition with DNN

## Description

This project implements a real-time face recognition system using a combination of Deep Neural Networks (DNN) for feature extraction and Random Forest for classification. The system captures live video from a webcam, detects faces, recognizes them, and marks attendance in a CSV file.

## Key Features

- **DeepFace Integration**: Uses the DeepFace library with VGG-Face model for robust face embedding extraction
- **Random Forest Classifier**: Employs a Random Forest model for efficient face recognition
- **Data Augmentation**: Includes image augmentation techniques to improve model robustness
- **Real-time Processing**: Captures and processes live video feed from webcam
- **Attendance System**: Automatically logs recognized faces with timestamps to a CSV file
- **OpenCV Integration**: For face detection and video frame processing

## Technical Components

1. **Face Embedding Extraction**: Uses VGG-Face DNN model to convert faces into 128-dimensional embeddings
2. **Classification**: Random Forest classifier trained on face embeddings
3. **Data Augmentation**: Image transformations to increase training data diversity
4. **Real-time Pipeline**: Continuous face detection and recognition from webcam feed
5. **Attendance Tracking**: CSV-based logging system with timestamps

## Requirements

- Python 3.x
- DeepFace
- OpenCV
- scikit-learn
- TensorFlow/Keras
- pandas
- numpy

## Usage

1. Organize your facial dataset in folders (one folder per person)
2. Run the notebook to train the Random Forest classifier
3. The system will automatically start webcam feed for real-time recognition
4. Press 'q' to quit the application

## Applications

- Automated attendance systems
- Security and surveillance
- Personalized user experiences
- Access control systems

The project demonstrates how traditional machine learning (Random Forest) can be effectively combined with deep learning (DNN embeddings) for efficient face recognition tasks.


