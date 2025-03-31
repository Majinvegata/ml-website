import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Kernel function for weights
def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.eye(m)  # Identity matrix for weights
    for j in range(m):
        diff = point - xmat[j]
        weights[j, j] = np.exp(np.dot(diff, diff.T) / (-2.0 * k**2))  # Corrected multiplication
    return weights
# Compute local weight
def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    W = np.linalg.inv(xmat.T @ (wei @ xmat) + 1e-5 * np.eye(xmat.shape[1])) @ (xmat.T @ (wei @ ymat))  
    return W
# Locally Weighted Regression
def localWeightRegression(xmat, ymat, k):
    m, n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = (xmat[i] @ localWeight(xmat[i], xmat, ymat, k)).item()  # Extract scalar
    return ypred
# Load dataset
data = pd.read_csv('9-dataset.csv')
# Extracting features and labels
bill = np.array(data['total_bill']).reshape(-1, 1)
tip = np.array(data['tip']).reshape(-1, 1)
# Preparing matrix
m = bill.shape[0]
one = np.ones((m, 1))  # Create column of ones
X = np.hstack((one, bill))  # Create X matrix with bias
# Set bandwidth 'k' and compute predictions
k = 0.5
ypred = localWeightRegression(X, tip, k)
# Sort values for plotting
SortIndex = np.argsort(X[:, 1], axis=0)
xsort = X[SortIndex]
# Plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(bill, tip, color='green', label='Data points')
ax.plot(xsort[:, 1], ypred[SortIndex], color='red', linewidth=3, label='LWR Prediction')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.legend()
plt.show()
