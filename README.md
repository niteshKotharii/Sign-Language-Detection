# Real-Time Sign Language Detection Using Mediapipe and Machine Learning

This project involves creating a real-time hand gesture recognition system for American Sign Language (ASL) using computer vision and machine learning models. The project uses `MediaPipe` to extract hand landmarks and classifies them using various machine learning models such as Random Forest, K-Nearest Neighbors (KNN), and Multi-Layer Perceptron (MLP). The final model is deployed to predict sign language characters based on real-time camera input.

<img width="260" alt="image" src="https://github.com/user-attachments/assets/080d10b9-a10c-4585-983b-7ffa5d9fb9a6">

## Table of Contents
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Dataset Collection](#dataset-collection)
- [Model Training](#model-training)
- [Real-Time Prediction](#real-time-prediction)
- [Contributing](#contributing)

## Technologies Used
- Python
- OpenCV
- MediaPipe
- Scikit-learn
- Pickle
- Numpy

## Features
- Real-time hand gesture recognition using a webcam.
- Utilizes MediaPipe for hand landmark detection.
- Supports character prediction for American Sign Language (ASL) gestures.
- Multiple machine learning models including Random Forest, KNN, and MLP classifiers.
- Achieved model accuracy of over 90%.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/niteshKotharii/Sign-Language-Detection.git
    ```
2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the pre-trained model or train your own (see [Model Training](#model-training)).

## How to Use

1. **Data Collection**:
    - Run the data collection script to capture hand gesture images:
      ```bash
      python collect_data.py
      ```
    - This will allow you to collect data for different gesture classes by pressing "Q" to capture the images for each class.

2. **Model Training**:
    - After collecting the data, use the provided script to train the models:
      ```bash
      python train_model.py
      ```
    - This script trains RandomForest, KNN, and MLP classifiers and saves the best-performing model.

3. **Real-Time Prediction**:
    - After training or downloading the model, you can run the real-time hand gesture detection script:
      ```bash
      python predict_real_time.py
      ```
    - The script will activate your webcam and display the recognized ASL character in real-time.

## Dataset Collection

- **Number of Classes**: The project collects data for 26 classes (A-Z, corresponding to ASL alphabet gestures).
- **Data Storage**: Captured images are saved in a folder structure where each class has its own sub-folder.
- **Customization**: You can modify the `collect_data.py` script to collect more or fewer samples per class.

## Model Training

- The script `train_model.py` will:
    1. Load the dataset from the saved image files.
    2. Process the hand landmarks using `MediaPipe`.
    3. Train different machine learning models (Random Forest, KNN, and MLP).
    4. Evaluate the models and save the best one (by default, MLP is used).
    5. Save the trained model as `model.p` using Pickle.

- **Training Command**:
    ```bash
    python train_model.py
    ```

## Real-Time Prediction

The real-time prediction is powered by the trained model and `MediaPipe`. The system captures video from your webcam, processes each frame, and predicts the corresponding ASL character.

- **Run Command**:
    ```bash
    python predict_real_time.py
    ```

- **Output**: The output will be the predicted sign language character, which is displayed on the webcam feed as an overlay.

## Contributing

Feel free to fork this repository, create a branch, and submit a pull request. Any contributions to improve the project or add new features are welcome!
