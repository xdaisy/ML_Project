import numpy as np
import pandas as pd
import PCA
import pickle

data = pd.read_pickle("digits.pkl")
df = pd.DataFrame(PCA.getEigenVectors(data.values, PCA.calc_mean(data.values)))
df.to_pickle("PCA_eigen_digit.pkl")


