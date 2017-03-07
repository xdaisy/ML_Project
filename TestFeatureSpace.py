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

    # get error for data space
    error = []
    featureSpace = []
    reconstructError = sys.maxint
    k = 1
    # find smallest feature space to reduce data set
    while reconstructError > .10:
        print "k: " + str(k)
        featureSpace.append(k)
        newSpace, U = PCA.reduce(data_np, k)
        reconstructError = PCA.reconstruction_error(newSpace, data_np, mean, k)
        print "reconstr error: " + str(reconstructError)
        error.append(reconstructError)
        k += 1
    print "Smallest feature space size: " + str(k - 1)
    plt.plot(featureSpace, error, marker=".")
    plt.ylabel("Reconstruction Error")
    plt.xlabel("Size of Reduced Feature Space")
    plt.title("Size of Reduced Feature Space vs Reconstruction Error")
    plt.savefig("Error for PCA")

if __name__ == "__main__":
    main()