# Bayes Classifier for 2D Data

This repository contains the implementation of a Bayes classifier using multivariate Gaussian distributions to classify points based on the decision boundary between two classes,  𝑦=0 and y=1. The classifier is trained on a dataset and tested on another dataset, with the results saved in answers.csv.

This repository represents my code solution for a homework assignment in the MIT class 6.7900, Machine Learning. The assignment required implementing a Bayes classifier to solve classification problems using provided datasets. The approach and implementation demonstrate the application of statistical methods and machine learning principles discussed in the course.

## Files Overview:
train.csv: The dataset used to train the classifier. It contains two features (x1, x2) and a label (y), which takes values 0 or 1.

test.csv: The dataset of points to classify. It contains two features (x1, x2) for each point. The classifier predicts whether each point belongs to class 0 or class 1.

answers.csv: The output file that stores the predicted labels (predicted_y) for the points in test.csv.

get_parameters.py: Reads the data, computes the necessary parameters (mean, covariance) for both classes.

classify.py: uses parameters computed in get_paramters to classify data in test.csv.

## Methodology
The classifier assumes that the two classes 
𝑦=0 y=1 follow multivariate normal distributions with means 𝜇_1 and 𝜇_2, and covariances Σ_1 and Σ_2, respectively. The decision boundary is derived by comparing the log-likelihoods of each class, which results in a quadratic decision boundary when 
Σ_1≠Σ_2​ and a linear boundary when Σ_1=Σ_2. The parameter values calculated with get_parameters.py suggest that for this dataset, Σ_1=Σ_2 is a valid assumption.

## License

This project is licensed under the MIT License.

​
 .
