import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics

# Load dataset
names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Class']
dataset = pd.read_csv('7-dataset.csv', names=names)
dataset.columns = dataset.columns.str.strip()  # Remove any extra spaces

# Remove non-numeric rows and ensure correct data types
dataset = dataset[pd.to_numeric(dataset['Sepal_Length'], errors='coerce').notna()]
dataset.iloc[:, :-1] = dataset.iloc[:, :-1].astype(float)

X = dataset.iloc[:, :-1]

# Convert class labels to numerical values
label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = np.array([label.get(c.strip(), -1) for c in dataset['Class']])

# Plot configuration
plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])

# Real Plot
plt.subplot(1, 3, 1)
plt.title('Real')
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y])

# KMeans Clustering
kmeans_model = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
plt.subplot(1, 3, 2)
plt.title('KMeans')
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[kmeans_model.labels_])
print('The accuracy score of K-Means:', metrics.accuracy_score(y, kmeans_model.labels_))
print('The Confusion Matrix of K-Means:\n', metrics.confusion_matrix(y, kmeans_model.labels_))

# Gaussian Mixture Model (GMM) Clustering
gmm_model = GaussianMixture(n_components=3, random_state=0).fit(X)
y_cluster_gmm = gmm_model.predict(X)
plt.subplot(1, 3, 3)
plt.title('GMM Classification')
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm])
print('The accuracy score of GMM:', metrics.accuracy_score(y, y_cluster_gmm))
print('The Confusion Matrix of GMM:\n', metrics.confusion_matrix(y, y_cluster_gmm))

# Show Plots
plt.tight_layout()
plt.show()
