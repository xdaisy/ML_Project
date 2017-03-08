import pandas as pd
import numpy as np
import k_means
import sys
import spectral
import PCA
from matplotlib import pyplot as plt

def SSD(train, cluster_center, cluster_idx):
    n = train.shape[0]
    ss = np.zeros((cluster_center.shape[0], ))
    for i in range(n):
        cluster = cluster_idx[i]
        #diff = train[i, :] - cluster_center[cluster]
        ss[cluster] += k_means.distance(train[i, :], cluster_center[cluster])
    return np.sum(ss)

def main():
    train = pd.read_pickle("cluster.pkl")
    reduced_data = PCA.reduce(train.values, 50, PCA.getU("PCA_eigen_cluster.pkl").values, PCA.calc_mean(train.values))
    heterogeneity_k_means = []
    heterogeneity_spectral = []
    ks = range(1,51)
    spectral_laplacian = spectral.setup(train.values)
    for k in ks:
        print "k: " + str(k)
        bestSSD_k_means = sys.maxint
        bestSSD_spectral = sys.maxint
        spectral_eigen = spectral.computeEigen(spectral_laplacian, k)
        # do clustering 3 times for each k
        for i in range(5):
            print "i: " + str(i)
            print "k_means"
            cluster_center_k_means, cluster_idx_k_means = k_means.cluster(reduced_data, k)
            ssd_k_means = SSD(reduced_data, cluster_center_k_means, cluster_idx_k_means)
            if ssd_k_means < bestSSD_k_means:
                bestSSD_k_means = ssd_k_means
            print "Spectral"
            cluster_center_spectral, cluster_idx_spectral = spectral.cluster(spectral_eigen, k)
            ssd_spectral = SSD(spectral_eigen, cluster_center_spectral, cluster_idx_spectral)
            if ssd_spectral < bestSSD_spectral:
                bestSSD_spectral = ssd_spectral
        # append best ssd
        heterogeneity_k_means.append(bestSSD_k_means)
        heterogeneity_spectral.append(bestSSD_spectral)
    plt.figure(1)
    plt.plot(ks, heterogeneity_k_means, marker=".")
    plt.ylabel("Heterogeneity")
    plt.xlabel("k")
    plt.title("k vs Heterogeneity for k means")
    plt.xticks(np.arange(0, max(ks), 2.0))
    plt.savefig("heterogeneity_k_means_cluster.png")
    plt.figure(2)
    plt.plot(ks, heterogeneity_spectral, marker=".")
    plt.ylabel("Heterogeneity")
    plt.xlabel("k")
    plt.title("k vs Heterogeneity for spectral")
    plt.xticks(np.arange(0, max(ks), 2.0))
    plt.savefig("heterogeneity_spectral_cluster.png")

if __name__ == "__main__":
    main()
