import numpy as np
import pandas as pd
import PCA
from matplotlib import pyplot as plt
import sys

def main():
    # get dataset
    data = pd.read_pickle("tfidf_small.pkl")
    data_np = data.values

    # get mean
    mean = PCA.calc_mean(data_np)
    U = PCA.getEigenVectors(data_np, mean)

    # get error for data space
    error = []
    featureSpace = []
    reconstructError = sys.maxint
    k = 0
    # find smallest feature space to reduce data set
    while k < 20:
        print "k: " + str(k)
        k += 1
        newSpace, U = PCA.reduce(data_np, k, U, mean)
        reconstructError = PCA.reconstruction_error(newSpace, data_np, U, mean, k)
        print "reconstr error: " + str(reconstructError)
        error.append(reconstructError)
        featureSpace.append(k)
    print "Smallest feature space size: " + str(k - 1)
    plt.plot(featureSpace, error, marker=".")
    plt.ylabel("Reconstruction Error")
    plt.xlabel("Size of Reduced Feature Space")
    plt.title("Size of Reduced Feature Space vs Reconstruction Error")
    plt.savefig("Error for PCA")

if __name__ == "__main__":
    main()
