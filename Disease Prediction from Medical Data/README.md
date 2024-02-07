# Multimodal Heart Disease Prediction

## Project Description
This project aims to predict heart disease using a multimodal approach that combines ECG image data and structured health data. By leveraging the strengths of both image processing and traditional structured data analysis, we aim to create a more accurate and robust model for heart disease prediction. The project utilizes two main datasets: ECG images from a specified Kaggle dataset and structured heart disease data from the UCI Machine Learning Repository.

## Datasets
- [**ECG Images Dataset**](https://www.kaggle.com/datasets/erhmrai/ecg-image-data): Sourced from Kaggle, this dataset comprises ECG images used to extract features relevant to heart health.
- [**Heart Disease Dataset**](https://archive.ics.uci.edu/dataset/45/heart+disease): Obtained from the UCI Machine Learning Repository, including structured data on patient demographics, blood tests, and heart health metrics.

## Data Preprocessing Steps
1. **Image Preprocessing**:
   - ECG images are resized to a uniform dimension.
   - Principal Component Analysis (PCA) is applied to reduce the dimensionality of flattened ECG images, retaining the top 50 components for model input.

2. **Structured Data Preprocessing**:
   - Missing values are imputed using a Custom KNN Imputer for both numerical and categorical data.
   - Categorical variables (e.g., 'slope', 'restecg', 'cp', 'thal') are one-hot encoded.
   - Numerical features are standardized using StandardScaler.

3. **Data Fusion**:
   - Features from processed ECG images and structured data are concatenated to form a single feature vector for each patient.

## Model Architecture
The project employs a multimodal neural network with two branches:
- A **structured data branch** processes numerical and categorical features through dense layers.
- An **image processing branch** uses convolutional layers to extract features from ECG images.
These branches are then concatenated and followed by dense layers, culminating in a binary classification output for heart disease prediction.

## Cross-Validation
Cross-validation is performed to evaluate the model's performance, ensuring robustness and generalizability.

## Results
The multimodal approach demonstrates an improvement in prediction accuracy over models using only structured data or image data, showcasing the benefit of integrating diverse data types.


