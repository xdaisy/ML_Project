import pandas as pd
import numpy as np
import k_means
import sys
from matplotlib import pyplot as plt

def SSD(train, cluster_center, cluster_idk):
    n = train.shape[0]
    ss = np.zeros((cluster_center.shape[0], ))
    for i in range(n):
        cluster = cluster_idk[i]
        diff = train[i, :] - cluster_center[cluster]
        ss[cluster] += np.dot(diff.T, diff)
    return np.sum(ss)

def main():
    train = pd.read_pickle("tfidf_medium.pkl")
    heterogeneity = []
    ks = range(1, 20)
    for k in ks:
        bestSSD = sys.maxint
        # do clustering 3 times for each k
        for i in range(5):
            cluster_center, cluster_idk = k_means.cluster(train.values, k)
            ssd = SSD(train.values, cluster_center, cluster_idk)
            if ssd < bestSSD:
                bestSSD = ssd
        # append best ssd
        heterogeneity.append(ssd)
    plt.plot(ks, heterogeneity, marker=".")
    plt.ylabel("Heterogeneity")
    plt.xlabel("k")
    plt.title("k vs Heterogeneity")
    plt.xticks(np.arange(0, max(ks), 2.0))
    plt.savefig("heterogeneity.png")

if __name__ == "__main__":
    main()