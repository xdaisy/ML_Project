import numpy as np

def reduce(values, k):
    mean = calc_mean(values)
    for i in range(values.shape[1]):
        values[:, i] = values[:, i] - mean[i]
    cov = np.cov(values)
    U, S, V = np.linalg.svd(cov)
    result = U[:, :k]
    #for i in range(k):
    #    result[:, i] = result[:, i] / np.linalg.norm(result[:, i])
    z = np.zeros((values.shape[0], k))
    for i in range(values.shape[0]):
        x_i = values[i, :]
        for j in range(k):
            print x_i.shape
            print mean.shape
            print U[:, k].shape
            print U.shape
            z[i, k] = np.dot((x_i - mean), U[:, k])
    return z, U

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
            res += original[i, j] * eigen_vectors[:, j]
        reconstructed[i, :] = mean + res
    error = calc_error(reconstructed, original)
    return error/ baseline