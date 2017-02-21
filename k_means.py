import numpy as np
import pandas as pd
import parser
import random
import math

# get parser
p = parser.Parser("tfidf.txt")
# parse training data

#parse testing data

# find k mean clusters
def cluster(train):
    k = 3
    cluster = np.zeros((3, train.shape[1]))
    # initialize cluster mean
    for i in range(k):
        # get random document from training data
        row = random.randint(0, train.shape[0] - 1)
        for j in range(train.shape[1]):
            cluster[i, j] = train[row, j]
    cluster_idx = np.zeros((train.shape[0]))
    iterations = 100
    # get cluster mean
    for i in range(iterations):
        clust = 0
        vector = train[i, :]
        minDist = distance(cluster[0, :], vector)
        for j in range(1, 3):
            dist = distance(cluster[j, :], vector)
            if (dist < minDist)
                clust = j
                minDist = dist


# compute distance
def distance(vector1, vector2):
    diff = vector2 - vector1
    return math.sqrt(np.dot(diff.T, diff))