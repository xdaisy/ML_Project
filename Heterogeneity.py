mport pandas as pd
import numpy as np
import k_means
from matplotlib import pyplot as plt

def SSD(train, cluster_center, cluster_idk):
    n = train.shape[0]
    sum = 0.0
    for i in range(n):
        cluster = cluster_idk[i]
        diff = train[i, :] - cluster_center[cluster]
        sum += np.dot(diff.T, diff)
    return sum

def main():
    train = pd.read_pickle("tfidf_medium.pkl")
    heterogeneity = []
    ks = range(1, 21)
    for k in ks:
        cluster_center, cluster_idk = k_means.cluster(train.values, k)
        h = SSD(train.values, cluster_center, cluster_idk)
        heterogeneity.append(h)
    plt.plot(ks, heterogeneity, marker=".")
    plt.ylabel("Heterogeneity")
    plt.xlabel("k")
    plt.title("k vs Heterogeneity")
    plt.xticks(np.arange(min(ks), max(ks), 2.0))
    plt.savefig("heterogeneity.png")

if __name__ == "__main__":
    main()
