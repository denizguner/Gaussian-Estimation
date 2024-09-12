import numpy as np
import pandas as pd

# Load test.csv
test_data = pd.read_csv('test.csv')
X_test = test_data[['x1', 'x2']].values

# Load the parameters from the training phase
mu_1 = np.array([-0.01226046, -0.03070837])  # Mean vector for y=0
mu_2 = np.array([5.08493872, 0.09534023])    # Mean vector for y=1

# Covariance matrix (assuming they are equal as per simplification)
Sigma = np.array([[1.00636175, 0.06389754],
                  [0.06389754, 1.13140123]])  # Common covariance matrix

# Prior probabilities
P_y0 = 0.5  # Assuming equal priors
P_y1 = 0.5

# Compute the inverse of the covariance matrix
Sigma_inv = np.linalg.inv(Sigma)

# Precompute terms for the discriminant function
mu_diff = mu_1 - mu_2
mu_1_term = mu_1.T @ Sigma_inv @ mu_1
mu_2_term = mu_2.T @ Sigma_inv @ mu_2
log_prior_ratio = np.log(P_y0 / P_y1)

# Threshold for the decision boundary
threshold = 0.5 * (mu_1_term - mu_2_term) + log_prior_ratio

# Classify each point in test.csv
y_pred = []
for x in X_test:
    discriminant_value = x.T @ Sigma_inv @ mu_diff
    if discriminant_value >= threshold:
        y_pred.append(0)  # Classify as y = 0
    else:
        y_pred.append(1)  # Classify as y = 1

# Add the predicted labels to the DataFrame
test_data['predicted_y'] = y_pred

# Write the classification results to a file called answers.csv
test_data[['x1', 'x2', 'predicted_y']].to_csv('answers.csv', index=False)

print("Results saved to answers.csv")