# Heart Disease Prediction using Machine Learning

This repository contains code for predicting heart disease using machine learning models. It demonstrates the process of feature engineering, preprocessing, model development, evaluation, and comparison.

## Dataset

The dataset used for this project is stored in a file named `heart.csv`. It contains various features related to heart health and the target variable, 'HeartDisease'. The dataset is loaded using pandas library.

## Feature Engineering and Preprocessing

Categorical variables are converted to numerical using one-hot encoding. The dataset is split into features (X) and the target variable (y). The data is then split into training and testing sets using the train_test_split function. Feature normalization is performed using StandardScaler.

## Machine Learning Models

The following machine learning models are trained and evaluated:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Logistic Regression

## Model Evaluation

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score

## Model Comparison

A comparison is made among the models based on their evaluation metrics. The best-performing algorithm is identified and its accuracy score is reported.

## Results

The results of model comparison, including accuracy, precision, recall, and F1-Score, are stored in a dataframe and printed. The best model and its accuracy score are also displayed.

## Model Comparison - Accuracy

A bar plot is generated to visualize the accuracy scores of the different models using matplotlib.

## Instructions

To run the code, follow these steps:
1. Clone the repository or download the code files.
2. Install the required libraries: pandas, scikit-learn.
3. Ensure that the 'heart.csv' file is present in the same directory as the code.
4. Run the code and observe the results.

Feel free to explore the code, modify it, and experiment with different models or evaluation metrics.

