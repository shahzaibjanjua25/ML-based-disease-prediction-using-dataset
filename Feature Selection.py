# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('heart.csv')

# 1. Feature Engineering and Preprocessing
# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

# Split the dataset into features (X) and target variable (y)
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Model Development
# Initialize and train different machine learning models
knn_model = KNeighborsClassifier()
svm_model = SVC()
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()
lr_model = LogisticRegression()

# Fit the models to the training data
knn_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# 3. Model Evaluation
# Make predictions on the test set
knn_pred = knn_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Calculate evaluation metrics for each model
knn_accuracy = accuracy_score(y_test, knn_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)

knn_precision = precision_score(y_test, knn_pred)
svm_precision = precision_score(y_test, svm_pred)
dt_precision = precision_score(y_test, dt_pred)
rf_precision = precision_score(y_test, rf_pred)
lr_precision = precision_score(y_test, lr_pred)

knn_recall = recall_score(y_test, knn_pred)
svm_recall = recall_score(y_test, svm_pred)
dt_recall = recall_score(y_test, dt_pred)
rf_recall = recall_score(y_test, rf_pred)
lr_recall = recall_score(y_test, lr_pred)

knn_f1 = f1_score(y_test, knn_pred)
svm_f1 = f1_score(y_test, svm_pred)
dt_f1 = f1_score(y_test, dt_pred)
rf_f1 = f1_score(y_test, rf_pred)
lr_f1 = f1_score(y_test, lr_pred)

# 4. Model Comparison
# Create a dataframe to compare the performance of the models
results = pd.DataFrame({
    'Model': ['KNN', 'SVM', 'Decision Tree', 'Random Forest', 'Logistic Regression'],
    'Accuracy': [knn_accuracy, svm_accuracy, dt_accuracy, rf_accuracy, lr_accuracy],
    'Precision': [knn_precision, svm_precision, dt_precision, rf_precision, lr_precision],
    'Recall': [knn_recall, svm_recall, dt_recall, rf_recall, lr_recall],
    'F1-Score': [knn_f1, svm_f1, dt_f1, rf_f1, lr_f1]
})

# Print the results
print(results)

# Identify the best-performing algorithm based on the evaluation metrics
best_model = results.loc[results['Accuracy'].idxmax(), 'Model']
best_accuracy = results.loc[results['Accuracy'].idxmax(), 'Accuracy']
print('Best Model:', best_model)
print('Accuracy:', best_accuracy)
import matplotlib.pyplot as plt

# Plot the accuracy scores
plt.figure(figsize=(10, 6))
plt.bar(results['Model'], results['Accuracy'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison - Accuracy')
plt.ylim([0, 1])
plt.xticks(rotation=45)
plt.show()
