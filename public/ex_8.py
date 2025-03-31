import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
# Define column names
column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# Read dataset
dataset = pd.read_csv('8-dataset.csv', names=column_names, header=None)
# Print dataset preview
print("\nFirst 5 rows of dataset:")
print(dataset.head())
# Convert feature columns to numeric (force conversion)
X = dataset.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
# Check for non-numeric values
if X.isnull().any().any():
    print("\nWarning: Some non-numeric values were found and converted to NaN. Fixing dataset...")
    print(X[X.isnull().any(axis=1)])  # Show affected rows
    X = X.dropna()  # Remove problematic rows
    dataset = dataset.loc[X.index]  # Keep corresponding labels in sync
# Extract labels (y)
y = dataset.iloc[:, -1]
# Encode class labels
label_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = y.map(label_mapping)
# Check if all labels were mapped correctly
if y.isnull().any():
    raise ValueError("Some class labels couldn't be mapped. Check for typos or unexpected values.")
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
# Train KNN model
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
# Predict on test data
y_pred = classifier.predict(X_test)
# Display results
print("\n" + "-" * 75)
print(f'{"Original Label":<20} {"Predicted Label":<20} {"Correct/Wrong"}')
print("-" * 75)
for actual, predicted in zip(y_test, y_pred):
    correct = "Correct" if actual == predicted else "Wrong"
    print(f'{actual:<20} {predicted:<20} {correct}')
print("-" * 75)
print("\nConfusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
print("-" * 75)
print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))
print("-" * 75)
print(f'Accuracy of the classifier: {metrics.accuracy_score(y_test, y_pred):.2f}')
print("-" * 75)
