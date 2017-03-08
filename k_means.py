import numpy as np
import pandas as pd
import Parser
import random
import math
import sys
from scipy import spatial
import PCA

# find k mean clusters
def cluster(train, k):
    dim = train.shape
    cluster_center = np.zeros((k, dim[1]))
    # initialize cluster mean
    chosen_rows = []
    for i in range(k):
        # get random document from training data
        row = random.randint(0, dim[0] - 1)
        while row in chosen_rows:
            row = random.randint(0, dim[0] - 1)
        chosen_rows.append(row)
        cluster_center[i] = train[row]
    # Nx1 array for keeping track of which point is in what cluster
    cluster_idx = np.zeros((dim[0],), dtype=np.int)
    iterations = 0
    # get cluster mean
    while True:
        #print iterations
        iterations += 1
        old_cluster_idx = np.copy(cluster_idx)
        # for each document, find cluster it belongs to
        for i in range(dim[0]):
            clust = 0
            doc = train[i, :]
            minDist = distance(cluster_center[0, :], doc)
            # find closest cluster
            for j in range(1, k):
                dist = distance(cluster_center[j, :], doc)
                if dist < minDist:
                    clust = j
                    minDist = dist
            cluster_idx[i] = clust

        # initialize to zero
        meanCluster = np.zeros((k, dim[1]))
        # kx1 array for keeping track of number of points in each cluster
        cluster_count = np.zeros((k,), dtype=np.int)
        # add up values for each cluster
        for i in range(dim[0]):
            meanCluster[cluster_idx[i], :] += train[i, :]
            cluster_count[cluster_idx[i]] += 1
        # find average
        converged = True
        for i in range(k):
            # get mean of words
            new_mean = meanCluster[i, :] / cluster_count[i]
            # compute distance between current mean and next mean
            #diff = distance(cluster_center[i, :], new_mean)
            # put into cluster numpy
            cluster_center[i, :] = np.copy(new_mean)
            #if diff > 0.0000001:
             #   converged = False
        # if converged, break out of loop
        #if converged:
        #    break
        if np.all(cluster_idx == old_cluster_idx):
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
    #diff = vector2 - vector1
    #return math.sqrt(np.dot(diff.T, diff))
    return spatial.distance.cosine(vector1, vector2)

def too_close(row, chosen_rows, train):
    for i in range(len(chosen_rows)):
        print distance(train[row], train[chosen_rows[i]])
        if distance(train[row], train[chosen_rows[i]]) < 15:
            print "too close"
            return True
    print "good"
    return False

def main():
    global group_num
    # get parser
    k = int(sys.argv[1])
    print 1
    print 2
    train = pd.read_pickle("cluster.pkl")
    reduced_train = PCA.reduce(train.values, 50, PCA.getU("PCA_eigen_cluster.pkl").values, PCA.calc_mean(train.values))
    print 3
    cluster_center, cluster_idx = cluster(reduced_train, k)
    print 4
    print cluster_center
    print cluster_center.shape
    print cluster_idx
    print 5
    articles = train.index.values
    groupings = {}
    for i in range(k):
        group_num = i
        b = np.apply_along_axis(isInGroup, 0, cluster_idx)
        groupings[i] = articles[b]
    print(groupings)
    for key in groupings:
        print groupings[key].shape


def isInGroup(a):
    global group_num
    return a == group_num

if __name__ == "__main__":
    main()
