# Credit Score Classification Project

## Overview
This project aims to build a predictive model using the Credit Score Classification dataset, available on Kaggle. Our goal is to classify individuals based on their credit score, which is crucial for financial institutions to make informed lending decisions. The dataset includes various features like age, income, loan amount, and history, which are indicative of an individual's creditworthiness.

## Dataset
The dataset for this project can be found at [Kaggle: Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification?select=train.csv). It includes a training set with features and labels for developing the model and a test set for evaluating its performance.

## Models Used
We employed several machine learning models to address this classification problem:

- `BaggingClassifier`: Enhances stability and accuracy by aggregating predictions from multiple models.
- `ExtraTreesClassifier`: Implements a meta estimator that fits a number of randomized decision trees to improve predictive accuracy.
- `RandomForestClassifier`: Offers a robust method via averaging predictions from multiple decision trees.
- `HistGradientBoostingClassifier`: Utilizes a histogram-based approach for gradient boosting, efficient for large datasets.
- `XGBClassifier`: Employs gradient boosting framework XGBoost for high efficiency and performance.

A `StackingClassifier` combines these models, using their predictions as input for a final estimator to improve prediction accuracy.

## Implementation
The project employs a `StackingClassifier` for model ensembling, leveraging the strengths of individual classifiers. The base models include a variety of classifiers, each contributing unique insights into the data. The final predictions are made based on the combined output of these models, processed through a meta-model, which in this case is trained on the predictions of the base models.

### Feature Engineering
We transformed continuous input variables into discrete bins for the `HistGradientBoostingClassifier`, enhancing model training speed and efficiency.

### Handling Missing Values
Models were chosen and configured to handle missing values effectively, ensuring robust performance across the dataset.

### Model Training and Evaluation
Each model was trained on the dataset, with performance evaluated using standard classification metrics. The stacking approach allowed us to capitalize on the diverse strengths of each base model, leading to improved overall accuracy.

## Dependencies
- Python 3.8+
- scikit-learn
- xgboost
- pandas
- numpy
