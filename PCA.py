import numpy as np

def reduce(values, k):
    mean = calc_mean(values)
    for i in range(values.shape[1]):
        values[:, i] = values[:, i] - mean[i]
    cov = np.cov(values)
    U, S, V = np.linalg.svd(cov)
    result = U[:, :k]
    for i in range(k):
        result[:, i] = result[:, i] / np.linalg.norm(result[:, i])
    return result

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

def reconstruction_error(reduced, original, mean, k):
    reconstructed = np.zeros(original.shape[0], original.shape[1])
    baseline = calc_error(reconstructed, original)
    for i in range(original.shape[0]):
        res = 0
        for j in range(k):
            z_ij = np.dot(original[i, :] - mean, reduced[j])
            res += z_ij * reduced[j]
        reconstructed[i, :] = mean + res
    error = calc_error(reconstructed, original)
    return error/ baseline