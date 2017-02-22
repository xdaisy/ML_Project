import numpy as np
import pandas as pd
import parser
import random
import math
import sys

# find k mean clusters
def cluster(train, k):
    dim = train.shape
    cluster = np.zeros((3, dim[1]))
    # initialize cluster mean
    for i in range(k):
        # get random document from training data
        row = random.randint(0, dim[0] - 1)
        cluster[i, :] = train[row, :]
        #for j in range(dim[1]):
        #    cluster[i, j] = train[row, j]
    cluster_idx = np.zeros((dim[0],))
    cluster_count = np.zeros((3,))
    iterations = 100
    # get cluster mean
    for it in range(iterations):
        # for each document, find cluster
        for i in range(dim[0]):
            clust = 0
            doc = train[i, :]
            minDist = distance(cluster[0, :], doc)
            # find closest cluster
            for j in range(1, k):
                dist = distance(cluster[j, :], doc)
                if dist < minDist:
                    clust = j
                    minDist = dist
            cluster_idx[i] = clust
            cluster_count[clust] += 1
        # calculate mean cluster
        meanCluster = np.zeros((3, dim.shape[1]))
        # add up values for each cluster
        for i in range(dim[0]):
            meanCluster[cluster_idx[i], :] += train[i, :]
        # find average
        converged = True
        for i in range(k):
            mean = meanCluster[i, :] / cluster_count[i]
            diff = distance(cluster[i, :], mean)
            cluster[i, :] = mean
            if diff > 0.00001:
                converged = False
        if converged:
            break
# compute distance
def distance(vector1, vector2):
    diff = vector2 - vector1
    return math.sqrt(np.dot(diff.T, diff))

def main():
    # get parser
    p_train = parser.Parser("tfidf_train.txt")
    # parse training data
    train = p_train.parse()
    cluster(train, sys.argv[1])
    # parse testing data


if __name__ == "__init__":
    main()