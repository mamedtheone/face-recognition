# FaceAuth AI

FaceAuth AI is a capstone-ready facial authentication project that lets users register with their face, train a machine learning model, and log in without a password. The project combines OpenCV for image processing, scikit-learn for classification, and Streamlit for the user interface.

## Problem Statement

Traditional password systems are easy to forget and vulnerable to theft. This project explores a simple biometric alternative by using facial recognition to decide whether a user should be granted access.

## Features

- Face registration with webcam snapshots through Streamlit or OpenCV.
- Face detection and preprocessing using Haar Cascade.
- Feature engineering by flattening normalized 64x64 face images.
- K-Nearest Neighbors classifier for user recognition.
- Login flow with probability and distance threshold checks.
- Evaluation using accuracy, confusion matrix, and classification report.

## Project Structure

```text
faceauth-ai/
├── app/
│   ├── main.py
│   ├── pages/
│   │   ├── login.py
│   │   └── register.py
│   └── utils.py
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data_collection.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── REPORT.md
├── config.py
├── README.md
└── requirements.txt
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app/main.py
```

## How It Works

### 1. Registration

- Enter a user name or ID on the Register page.
- Capture 5 to 10 images with the camera widget.
- Each image is stored in `data/raw/<user_id>/`.
- A preprocessed version is saved to `data/processed/<user_id>/`.

### 2. Preprocessing

- Detect the face using OpenCV Haar Cascade.
- Crop the largest detected face.
- Convert to grayscale.
- Resize to 64x64.
- Normalize pixel values to the range `[0, 1]`.

### 3. Training

- Load processed images.
- Flatten each image into a 1D vector.
- Split the dataset into train and test sets.
- Train a KNN classifier with distance weighting.
- Save the trained model to `models/face_model.pkl`.

### 4. Login

- Capture a new face image on the Login page.
- Apply the same preprocessing steps.
- Predict the closest registered user.
- Grant access only if the predicted probability is above the probability threshold and the average KNN distance is below the learned distance threshold.

## Model Choice

K-Nearest Neighbors is a good fit for this capstone because it is simple to explain, works well for small datasets, and makes distance-based thresholding straightforward.

## Evaluation

The training script stores:

- Accuracy
- Confusion matrix
- Classification report

These metrics are shown in the app home page after training.

## Screenshots

Add local screenshots after your first run for the final submission:

- Register page capturing multiple samples
- Login page showing access granted or denied
- Home page showing accuracy and confusion matrix

## Testing

Run:

```bash
pytest
```

The provided test suite checks preprocessing, model training, and prediction on synthetic face-like image data.

## Future Improvements

- Replace Haar Cascade with a stronger detector such as MTCNN or MediaPipe.
- Use facial embeddings instead of raw pixel features.
- Add persistent user management and audit logs.
- Support multiple login attempts with anti-spoofing checks.
