import pandas as pd
import numpy as np
import spectral
import PCA
import k_means
import random
from matplotlib import pyplot as plt

def computeMistakes(cluster_indx, y, k):
    mistakes = 0
    # count the number of times a label appear in cluster i
    for i in range(k):
        cluster_map = {}
        count = 0
        for j in range(cluster_indx.shape[0]):
            if spectral_indx[j] != i:
                continue
            count += 1
            if y[j] not in cluster_map.keys():
                cluster_map[y[j]] = 1
            else:
                cluster_map[y[j]] += 1
        label = ""
        max_label_count = 0
        # find label with most count
        for key in cluster_map.keys():
            if cluster_map[key] > max_label_count:
                label = key
                max_label_count = cluster_map[key]
        for key in cluster_map.keys():
            if key != label:
                mistakes += cluster_map[key]
    return mistakes

data = pd.read_pickle("cluster.pkl")
reduced_data = PCA.reduce(data.values, 50, PCA.getU("PCA_eigen_cluster.pkl").values, PCA.calc_mean(data.values))
y = []
labels_file = open("cluster_labels.txt")
for line in labels_file:
    y.append(line)
y = np.asarray(y)
affinity =spectral.computeAffinity(data.values)
spectral_mistakes = []
kmean_mistakes = []
kmean_reduced_mistakes = []
# get mistakes for spectral, k means, k means reduced for k = 2,...,50
for k in range(2, 51):
    spectral_centers, spectral_indx = spectral.cluster(spectral.computeEigen(spectral.computeLaplacian(affinity, spectral.computeDegree(affinity)), k), k)
    s_mistakes = computeMistakes(spectral_indx, y, k)
    spectral_mistakes.append(s_mistakes)
    kmeans_centers, kmeans_indx = k_means.cluster(data.values, k)
    k_mistakes = computeMistakes(kmeans_indx, y, k)
    kmean_mistakes.append(k_mistakes)
    kmeans_reduced_centers, kmeans_reduced_indx = k_means.cluster(reduced_data, k)
    kr_mistakes = computeMistakes(kmeans_reduced_indx, y, k)
    kmean_reduced_mistakes.append(kr_mistakes)

# graph spectral mistakes
"""plt.plot(range(2, 51), spectral_mistakes, marker=".")
plt.ylabel("Mistakes")
plt.xlabel("k")
plt.title("k vs Mistakes (Spectral)")
plt.savefig("Mistakes cluster Spectral.png")

# graph k means mistakes
plt.plot(range(2, 51), kmean_mistakes, marker=".")
plt.ylabel("Mistakes")
plt.xlabel("k")
plt.title("k vs Mistakes (K Means)")
plt.savefig("Mistakes cluster K Means.png")

plt.plot(range(2, 51), kmean_reduced_mistakes, marker=".")
plt.ylabel("Mistakes")
plt.xlabel("k")
plt.title("k vs Mistakes (K Means DR)")
plt.savefig("Mistakes cluster K Means DR.png")"""

plt.plot(range(2, 51), spectral_mistakes, marker="o")
plt.plot(range(2, 51), kmean_mistakes, marker="s")
plt.plot(range(2, 51), kmean_reduced_mistakes, marker="*")
plt.legend(['Spectral', 'K Means', 'K Means with PCA'], loc='upper right')
plt.title("k vs Mistakes")
plt.savefig("Mistakes.png")