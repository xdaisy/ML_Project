import pandas as pd
import numpy as np
import spectral

data = pd.read_pickle("cluster.pkl")
y = []
labels_file = open("cluster_labels.txt")
for line in labels_file:
    y.append(line)
y = np.asarray(y)
affinity =spectral.computeAffinity(data.values)
spectral_centers, spectral_indx = spectral.cluster(spectral.computeEigen(spectral.computeLaplacian(affinity, spectral.computeDegree(affinity)), 3), 3)
mistakes = 0
cluster_map = {}
for i in range(y.shape[0]):
    if y[i] not in cluster_map:
        cluster_map[y[i]] = spectral_indx[i]
    elif spectral_indx[i] != cluster_map[y[i]]:
        mistakes += 1
