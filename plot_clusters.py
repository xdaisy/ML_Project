import numpy as np
import pandas as pd
import k_means
import spectral
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    try:
        k = int(sys.argv[1])
    except IndexError:
        print "Usage: python plot_clusters.py <k>"
    colors = ['#000000', '#00FF00', '#0000FF', '#FF0000', '#01FFFE', '#FFA6FE', '#FFDB66', '#006401', 
'#010067', '#95003A', '#007DB5', '#FF00F6', '#FFEEE8', '#774D00', '#90FB92', '#0076FF', '#D5FF00',
'#FF937E', '#6A826C', '#FF029D', '#FE8900', '#7A4782', '#7E2DD2', '#85A900', '#FF0056', '#A42400',
'#00AE7E', '#683D3B', '#BDC6FF', '#263400', '#BDD393', '#00B917', '#9E008E', '#001544', '#C28C9F',
'#FF74A3', '#01D0FF', '#004754', '#E56FFE', '#788231', '#0E4CA1', '#91D0CB', '#BE9970', '#968AE8', 
'#BB8800', '#43002C', '#DEFF74', '#00FFC6', '#FFE502', '#620E00', '#008F9C', '#98FF52', '#7544B1', 
'#B500FF', '#00FF78', '#FF6E41', '#005F39', '#6B6882', '#5FAD4E', '#A75740', '#A5FFD2', '#FFB167', 
'#009BFF', '#E85EBE']
    train = pd.read_pickle("tfidf_medium_large.pkl")    
    PCA_eigens = spectral.computeEigen(np.cov(train.values), 3)

    cluster_centers, cluster_idx = k_means.cluster(train.values, k)
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Clusters using k means using k = " + str(k))
    for i in range(0, PCA_eigens.shape[0], 10):
        ax.scatter(PCA_eigens[i, 0], PCA_eigens[i, 1], PCA_eigens[i, 2], c=colors[cluster_idx[i] % len(colors)])
    plt.savefig("cluster_k_means_ml3D.png")

    laplac = spectral.setup(train)
    eig = spectral.computeEigen(laplac, k)
    cluster_centers, cluster_idx = spectral.cluster(eig, k)
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Clusters using spectral clustering using k = " + str(k))
    for i in range(0, PCA_eigens.shape[0], 10):
        ax.scatter(PCA_eigens[i, 0], PCA_eigens[i, 1], PCA_eigens[i, 2], c=colors[cluster_idx[i] % len(colors)])
    plt.savefig("cluster_spectral_ml3D.png")

if __name__ == "__main__":
    main()
