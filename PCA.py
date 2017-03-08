import numpy as np
import pandas as pd

def getEigenVectors(values, mean):
    for i in range(values.shape[1]):
        values[:, i] = values[:, i] - mean[i]
    cov = np.cov(values.T)
    print "got covariance"
    print cov.shape
    U, S, V = np.linalg.svd(cov)
    print "got eigen vectors"
    print U.shape
    return U

def getU(fileName):
    U = pd.read_pickle(fileName)
    return U

def reduce(values, k, U, mean):
    eigen_vectors = U[:, :k]
    # normalize eigen vectors
    for i in range(k):
        eigen_vectors[:, i] = eigen_vectors[:, i] / np.linalg.norm(eigen_vectors[:, i])
    z = np.zeros((values.shape[0], k))
    for i in range(values.shape[0]):
        x_i = values[i, :]
        for j in range(k):
            z[i, j] = np.dot((x_i - mean).T, U[:, j])
    return z, eigen_vectors

def calc_mean(values):
    mean = np.zeros(values.shape[1])
    for i in range(values.shape[1]):
        mean[i] = np.sum(values[:, i]) /len(values[:, i])
    return mean

def calc_error(reconstructed, original):
    res = 0
    for i in range(original.shape[0]):
        res += np.linalg.norm(reconstructed[i,:] - original[i, :])
    return res

def reconstruction_error(reduced, original, eigen_vectors, mean, k):
    reconstructed = np.zeros((original.shape[0], original.shape[1]))
    baseline = calc_error(reconstructed, original)
    for i in range(original.shape[0]):
        res = np.zeros((original.shape[1], ))
        for j in range(k):
            res += reduced[i, j] * eigen_vectors[:, j]
        reconstructed[i, :] = mean + res
    error = calc_error(reconstructed, original)
    return error #/ baseline
