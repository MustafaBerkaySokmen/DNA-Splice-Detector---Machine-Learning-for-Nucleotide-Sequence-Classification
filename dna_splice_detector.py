import numpy as np
import pandas as pd

# Use the full paths to the files
X = np.genfromtxt("hw01_data_points.csv", delimiter=",", dtype=str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter=",", dtype=int)

# STEP 3
# first 50000 data points should be included in train
# remaining 43925 data points should be included in test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    X_train = X[:50000]
    y_train = y[:50000]
    X_test = X[50000:]
    y_test = y[50000:]
    # your implementation ends above
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    unique_classes = np.unique(y)  # Find the unique classes in the dataset
    class_priors = np.zeros(len(unique_classes))  # Initialize an array to hold the prior probabilities
    for i, cls in enumerate(unique_classes):
        class_priors[i] = np.mean(y == cls)  # Calculate the prior probability for each class
    return class_priors

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)

# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    unique_classes = np.unique(y)  # Identify the unique classes
    D = X.shape[1]  # Number of positions in the sequence
    K = unique_classes.shape[0]  # Number of classes
    pAcd = np.zeros((K, D))
    pCcd = np.zeros((K, D))
    pGcd = np.zeros((K, D))
    pTcd = np.zeros((K, D))
    for c in unique_classes:
        for d in range(D):
            indices = np.where(y == c)[0]
            sequences = X[indices, d]
            pAcd[c-1, d] = np.sum(sequences == 'A') / len(sequences)
            pCcd[c-1, d] = np.sum(sequences == 'C') / len(sequences)
            pGcd[c-1, d] = np.sum(sequences == 'G') / len(sequences)
            pTcd[c-1, d] = np.sum(sequences == 'T') / len(sequences)
    return pAcd, pCcd, pGcd, pTcd

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)

# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    N, D = X.shape  # Number of data points and features
    K = class_priors.shape[0]  # Number of classes
    score_values = np.zeros((N, K))  # Logarithm of class priors to add to score
    log_class_priors = np.log(class_priors)
    for i in range(N):  # For each data point
        for c in range(K):  # For each class
            score = 0
            for d in range(D):  # For each feature/position
                nucleotide = X[i, d]
                if nucleotide == 'A':
                    score += np.log(pAcd[c, d])
                elif nucleotide == 'C':
                    score += np.log(pCcd[c, d])
                elif nucleotide == 'G':
                    score += np.log(pGcd[c, d])
                elif nucleotide == 'T':
                    score += np.log(pTcd[c, d])
            # Add log of class prior to the score for this class
            score_values[i, c] = score + log_class_priors[c]
    return score_values

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)
scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)

# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    N = scores.shape[0]
    K = np.unique(y_truth).shape[0]
    predicted_classes = np.argmax(scores, axis=1) + 1  # Assuming classes are 1-indexed
    confusion_matrix = np.zeros((K, K), dtype=int)
    for i in range(N):
        true_class = y_truth[i]
        pred_class = predicted_classes[i]
        confusion_matrix[true_class - 1, pred_class - 1] += 1
    return confusion_matrix

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)
confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
