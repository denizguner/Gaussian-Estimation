import numpy as np
import pandas as pd

# Load the data from the CSV file
data = pd.read_csv('train.csv')

# Split the data based on the value of y
data_0 = data[data['y'] == 0][['x1', 'x2']].to_numpy()
data_1 = data[data['y'] == 1][['x1', 'x2']].to_numpy()

# Calculate the mean vectors u1 and u2
u1 = np.mean(data_0, axis=0)
u2 = np.mean(data_1, axis=0)

# Calculate the covariance matrices Sigma1 and Sigma2
Sigma1 = np.cov(data_0, rowvar=False)
Sigma2 = np.cov(data_1, rowvar=False)

# Output the results
print("probability of p(y=0):", len(data_0)/(len(data_0)+len(data_1)))
print("probability of p(y=1):", len(data_1)/(len(data_0)+len(data_1)))
print("Mean vector for y=0 (u1):", u1)
print("Mean vector for y=1 (u2):", u2)
print("Covariance matrix for y=0 (Sigma1):\n", Sigma1)
print("Covariance matrix for y=1 (Sigma2):\n", Sigma2)