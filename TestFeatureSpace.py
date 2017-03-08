import numpy as np
import pandas as pd
import PCA
from matplotlib import pyplot as plt
import sys
import math

def main():
    # get dataset
    data = pd.read_pickle("cluster.pkl")
    data_np = data.values

    # get mean
    mean = PCA.calc_mean(data_np)
    U = PCA.getU("PCA_eigen_cluster.pkl")
    # get error for data space
    error = []
    featureSpace = []
    prevError = sys.maxint
    reconstructError = 0.0
    k = 0
    # find smallest feature space to reduce data set
    while math.fabs(prevError - reconstructError) > 0.00001:
        k += 1
        print "k: " + str(k)
        prevError = reconstructError
        newSpace, eigen_vectors = PCA.reduce(data_np, k, U.values, mean)
        reconstructError = PCA.reconstruction_error(newSpace, data_np, eigen_vectors, mean, k)
        print "reconstr error: " + str(reconstructError)
        error.append(reconstructError)
        featureSpace.append(k)
    print "Smallest feature space size: " + str(k)
    plt.plot(featureSpace, error, marker=".")
    plt.ylabel("Reconstruction Error")
    plt.xlabel("Size of Reduced Feature Space")
    plt.title("Size of Reduced Feature Space vs Reconstruction Error")
    plt.savefig("Error for PCA Cluster")

if __name__ == "__main__":
    main()
