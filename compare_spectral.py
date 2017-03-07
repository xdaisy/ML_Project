import pandas as pd
import numpy as np
import spectral
from sklearn import cluster

data = pd.read_pickle('tfidf_medium.pkl')
#affinity = cluster.SpectralClustering.fit(data.values)
sklearn_spectral = cluster.SpectralClustering(n_clusters=10)
sklearn_spectral.fit(data.values)
print sklearn_spectral.fit_predict(data.values)