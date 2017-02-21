import numpy as np
import pandas as pd
import parser
import math

# pass in text file name to constructor to parse training and test data
p = parser.Parser("tfidf_small.txt")
# parse training data

# parse testing data

def cluster(train):
    dim_train = train.shape
    # affinity matrix, is numpy array
    affinity = computeAffinity(train)
    # degree matrix, is numpy array
    degree = computeDegree(affinity)
    # laplcian matrix, is numpy array
    laplacian = computeLaplacian(affinity, degree)


def distance(vector1, vector2):
    diff = vector2 - vector1
    return math.sqrt(np.dot(diff.T, diff))


def computeAffinity(train):
    dim = train.shape
    affinity = np.zeros((dim[0], dim[0]))
    sigma = 10
    for i in range(dim):
        for j in range(dim):
            # if i == j, skip, don't compare to self
            if i == j:
                affinity[i, j] = 0
                continue
            affinity[i, j] = np.e**(-1 * distance(train[i, :], train[j,:]) / (2 * (sigma**2)))


def computeDegree(affinity):
    dim = affinity.shape
    degree_matrix = np.zeros((dim[0], dim[0]))
    for i in range(dim[0]):
        degree_matrix[i, i] = np.sum(affinity[i, :])
    return degree_matrix


def computeLaplacian(affinity, degree):
    return degree - affinity