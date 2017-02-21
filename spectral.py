import numpy as np
import pandas as pd
import parser
import math

# pass in text file name to constructor to parse training and test data
p = parser.Parser()
# parse training data

# parse testing data

def cluster(train, test):
    dim_train = train.shape
    # affinity matrix
    affinity_matrix = np.zeros((dim_train[0], dim_train[0]))
    degree_matrix = computeDegree(affinity_matrix)
    # go through each document
    sigma = 10
    # compute affinity matrix


def distance(vector1, vector2):
    v1 = vector1[1:]
    v2 = vector2[1:]
    diff = v2 - v1
    return math.sqrt(np.dot(diff.T, diff))

def computeAffinity(train):
    dim = train.shape
    affinity_matrix = np.zeros((dim[0], dim[0]))
    for i in range(dim):
        for j in range(dim):
            # if i == j, skip, don't compare to self
            if i == j:
                affinity_matrix[i, j] = 0
                continue
            affinity_matrix[i, j] = np.e**(-1 * distance(train[i, :], train[j,:]) / (2 * (sigma**2)))

def computeDegree(affin_matrix):
    dim = affin_matrix.shape
    degree_matrix = np.zeros((dim[0], dim[0]))
    for i in range(dim[0]):
        degree_matrix[i, i] = np.sum(affin_matrix[i, :])
    return degree_matrix