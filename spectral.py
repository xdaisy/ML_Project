import numpy as np
import pandas as pd
import Parser
import math
import sys
import k_means

def distance(vector1, vector2):
    diff = vector2 - vector1
    return math.sqrt(np.dot(diff.T, diff))


def computeAffinity(train):
    dim = train.shape
    affinity = np.zeros((dim[0], dim[0]))
    sigma = 10
    for i in range(dim[0]):
        for j in range(dim[0]):
            # if i == j, skip, don't compare to self
            if i == j:
                affinity[i, j] = 0
                continue
            affinity[i, j] = np.e**(-1 * distance(train[i, :], train[j,:]) / (2 * (sigma**2)))
    return affinity


def computeDegree(affinity):
    dim = affinity.shape
    degree_matrix = np.zeros((dim[0], dim[0]))
    for i in range(dim[0]):
        degree_matrix[i, i] = np.sum(affinity[i, :])
    return degree_matrix



def computeLaplacian(affinity, degree):
    return degree - affinity

# returns the first k eigen vectors of the laplacian matrix
def computeEigen(laplacian, k):
    U, S, V = np.linalg.svd(laplacian)

    result = U[:,:k]
    return result

def main():
    k = int(sys.argv[1])
    p = Parser.Parser("tfidf_small.txt")
    train = p.parse()
    # affinity matrix, is numpy array
    affinity = computeAffinity(train.values)
    # degree matrix, is numpy array
    degree = computeDegree(affinity)
    # laplcian matrix, is numpy array
    laplacian = computeLaplacian(affinity, degree)
    # compute eigen vectors
    U = computeEigen(laplacian, k)
    # run k_means
    cluster_center, cluster_idx = k_means.cluster(U, k)
    # display the data:
    print cluster_center
    print cluster_center.shape
    print cluster_idx

if __name__ == "__main__":
    main()