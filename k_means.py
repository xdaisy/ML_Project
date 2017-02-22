import numpy as np
import pandas as pd
import Parser
import random
import math
import sys

# find k mean clusters
def cluster(train, k):
    dim = train.shape
    cluster_center = np.zeros((3, dim[1]))
    train_np = train.values
    # initialize cluster mean
    for i in range(k):
        # get random document from training data
        row = random.randint(0, dim[0] - 1)
        cluster_center[i] = train_np[row]
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
            doc = train_np[i, :]
            minDist = distance(cluster_center[0, :], doc)
            # find closest cluster
            for j in range(1, k):
                dist = distance(cluster_center[j, :], doc)
                if dist < minDist:
                    clust = j
                    minDist = dist
            cluster_idx[i] = clust
            cluster_count[clust] += 1
        # calculate mean cluster
        meanCluster = np.zeros((3, dim[1]))
        # add up values for each cluster
        for i in range(dim[0]):
            meanCluster[cluster_idx[i], :] += train_np[i, :]
        # find average
        converged = True
        for i in range(k):
            # get mean of words
            mean = meanCluster[i, :] / cluster_count[i]
            # compute distance between current mean and next mean
            diff = distance(cluster_center[i, :], mean)
            # put into cluster numpy
            cluster_center[i, :] = np.copy(mean)
            if diff > 0.00001:
                converged = False
        # if converged, break out of loop
        if converged:
            break
    # return cluster mean, and the index of which doc corresponds to which index
    return cluster_center, cluster_idx

# compute the closest document to each document in test data set
def computeClosestDoc(train, test, cluster_center, cluster_idx):
    # keep track of the results for each test doc
    results = open("k_means_results.txt", "a")
    results.seek(0)
    results.truncate()
    dim = test.shape
    # go through each doc in test
    for i in range(dim[0]):
        closestCluster = findClosestCluster(test[i, :], cluster_center)
        bestDoc = findBestDoc(test[i, :], train, cluster_idx, closestCluster)
        results.write("test doc: " + test.index[i] + ", best doc: " + train.index[bestDoc] + "\n")


# find closest cluster
def findClosestCluster(x_i, cluster_center):
    cluster = 0
    minDist = distance(x_i, cluster_center[0, :])
    dim = cluster_center.shape
    for i in range(1, dim[0]):
        # compute distance
        dist = distance(x_i, cluster_center[i, :])
        # set i as closest cluster if dist < minDist
        if dist < minDist:
            cluster = i
            minDist = dist
    # return index of closest cluster
    return cluster

# find closest doc to x_i
def findBestDoc(x_i, train, cluster_idx, closestCluster):
    bestDoc = -1
    minDist = sys.maxsize
    dim = train.shape
    for i in range(dim[0]):
        # if not same cluster, skip
        if cluster_idx[i] != closestCluster:
            pass
        # compute distance between doc i in training set and current doc in test set
        dist = distance(x_i, train[i, :])
        # if have smaller dist, set bestDoc and minDist
        if dist < minDist:
            bestDoc = i
            minDist = dist
    # return index of best document
    return bestDoc

# compute distance
def distance(vector1, vector2):
    print vector1
    print vector2
    diff = vector2 - vector1
    return math.sqrt(np.dot(diff.T, diff))

def main():
    # get parser
    print 1
    p_train = Parser.Parser("tfidf_small.txt")
    print 2
    # parse training data
    train = p_train.parse()
    print 3
    cluster_center, cluster_idx = cluster(train, int(sys.argv[1]))
    print 4
    print cluster_center
    print cluster_idx
    print 5
    # parse testing data
    """
    p_test = parser.Parser("tfidf_test.txt")
    test = p_test.parse()
    computeClosestDoc(train, test, cluster_center, cluster_idx)
    """

if __name__ == "__main__":
    main()
