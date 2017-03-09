import pandas as pd
import numpy as np
import spectral
import PCA
import k_means
import sys

from matplotlib import pyplot as plt

def computeMistakes(cluster_indx, y, k):
    #print cluster_indx
    mistakes = 0
    # count the number of times a label appear in cluster i
    for i in range(k):
        cluster_map = {}
        count = 0
        for j in range(cluster_indx.shape[0]):
            if cluster_indx[j] != i:
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
        #print "cluster " + str(i)
        #print "label " + str(label).strip()
        #print "count " + str(count)
        #print "max label count " + str(max_label_count)
        #print "\n"
        mistakes += count - max_label_count
    #print "mistakes" + str(mistakes)
    #print y
    return mistakes

data = pd.read_pickle("cluster.pkl")
reduced_data_50 = PCA.reduce(data.values, 50, PCA.getU("PCA_eigen_cluster.pkl").values, PCA.calc_mean(data.values))[0]
reduced_data_10 = PCA.reduce(data.values, 2, PCA.getU("PCA_eigen_cluster.pkl").values, PCA.calc_mean(data.values))[0]
y = []
labels_file = open("cluster_labels.txt")
for line in labels_file:
    y.append(line)
y = np.asarray(y)
affinity =spectral.setup(data.values)
spectral_mistakes = []
kmean_mistakes = []
kmean_reduced_50_mistakes = []
kmean_reduced_10_mistakes = []
spectral_euc_mistakes = []
kmean_euc_mistakes = []
kmean_reduced_50_euc_mistakes = []
kmean_reduced_10_euc_mistakes = []
# get mistakes for spectral, k means, k means reduced for k = 2,...,50
for k in range(2,51):
    print k
    best = sys.maxint
    for i in range(5):
        spectral_centers, spectral_indx = spectral.cluster(spectral.computeEigen(affinity, k), k)
        s_mistakes = computeMistakes(spectral_indx, y, k)
        if s_mistakes < best:
            best = s_mistakes
    spectral_mistakes.append(best)
    best = sys.maxint
    for i in range(5):
        kmeans_centers, kmeans_indx = k_means.cluster(data.values, k)
        k_mistakes = computeMistakes(kmeans_indx, y, k)
        if k_mistakes < best:
            best = k_mistakes
    kmean_mistakes.append(best)
    best = sys.maxint
    for i in range(5):
        kmeans_reduced_50_centers, kmeans_reduced_50_indx = k_means.cluster(reduced_data_50, k)
        kr_50_mistakes = computeMistakes(kmeans_reduced_50_indx, y, k)
        if kr_50_mistakes < best:
            best = kr_50_mistakes
    kmean_reduced_50_mistakes.append(best)
    best = sys.maxint
    for i in range(5):
        kmeans_reduced_10_centers, kmeans_reduced_10_indx = k_means.cluster(reduced_data_10, k)
        kr_10_mistakes = computeMistakes(kmeans_reduced_10_indx, y, k)
        if kr_10_mistakes < best:
            best = kr_10_mistakes
    kmean_reduced_10_mistakes.append(best)
# graph spectral mistakes

plt.figure(1)
plt.plot(range(2, 51), spectral_mistakes, marker="o")
plt.plot(range(2, 51), kmean_mistakes, marker="s")
plt.plot(range(2, 51), kmean_reduced_50_mistakes, marker="*")
plt.plot(range(2, 51), kmean_reduced_10_mistakes, marker=".")
plt.legend(['Spectral', 'K Means', 'K Means with PCA = 50D', 'K Means with PCA = 2D'], loc='upper right')
plt.xlabel("k")
plt.ylabel("Mistakes")
plt.title("k vs Mistakes")
plt.savefig("Mistakes_Cluster_Cosine.png")
