# FaceAuth AI Report

## 1. Problem Statement

This project addresses a simple biometric authentication problem: instead of using a password, the system should recognize a registered user's face and decide whether to grant access. The goal is to build an end-to-end machine learning product, not just a standalone model. That means the solution must support data collection, preprocessing, training, evaluation, inference, and an interface that users can interact with.

## 2. Approach

The system was designed as a small but complete pipeline:

1. A user registers through the Streamlit app by entering a name or ID and capturing multiple face images.
2. The images are stored in a raw dataset folder.
3. Each image is processed with OpenCV Haar Cascade face detection.
4. The detected face is cropped, converted to grayscale, resized to 64x64, normalized, and flattened into a feature vector.
5. A supervised learning model is trained to map the feature vector to the correct user label.
6. During login, the same preprocessing steps are applied to a fresh face image and the trained model predicts the most likely user.
7. A confidence rule is applied before granting access so the system avoids obvious false positives.

This architecture keeps the project simple enough for a capstone while still reflecting the structure of a real ML application.

## 3. Model Used and Why

The selected model is **K-Nearest Neighbors (KNN)**. It was chosen for three main reasons:

- It is easy to implement and explain.
- It works well for small image datasets, which matches a classroom capstone scenario.
- It naturally supports distance-based matching, which makes it easier to implement a realistic access threshold.

Before training, each processed face image is flattened into a one-dimensional numeric vector. A `StandardScaler` is applied before KNN so pixel features are on a consistent scale. The classifier then predicts the user based on the closest stored examples.

## 4. Results

The training pipeline calculates:

- Accuracy
- Confusion matrix
- Classification report

The exact accuracy depends on the quality and quantity of captured images. In general, the system performs better when:

- Each user has at least 5 to 10 samples
- Lighting is reasonably consistent
- The face is clearly visible and centered
- Users are visually distinct

To reduce false acceptance, the login step combines two checks:

- The predicted class probability must exceed a minimum threshold.
- The average nearest-neighbor distance must stay below a learned distance threshold.

If either check fails, the system returns **Access Denied**.

## 5. Challenges and Issues

The main limitations of this version are tied to the simplicity of the approach:

- Haar Cascade can miss faces when lighting is poor or the pose changes.
- Raw pixel features are not as robust as learned face embeddings.
- A small dataset can lead to unstable accuracy.
- Browser-based camera capture in Streamlit is less real-time than a dedicated desktop login system.

Even with these limitations, the project still demonstrates the full machine learning workflow required for the capstone.

## 6. Improvements

This project can be improved in several ways:

- Use a modern face detector or face embedding model.
- Add anti-spoofing to prevent printed-photo attacks.
- Store metadata for each login attempt.
- Expand evaluation with more users and varied environments.
- Package the model and UI for easier deployment.

## 7. Conclusion

FaceAuth AI successfully demonstrates a functional AI-based login system with registration, training, evaluation, and authentication. It satisfies the capstone goal of building a complete ML product and provides a clear baseline that can be extended into a more advanced biometric system.
