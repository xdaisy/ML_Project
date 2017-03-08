import numpy as np
import pandas as pd
X = np.genfromtxt("digit.txt")
df = pd.Dataframe(X)
pd.to_pickle("digits.pkl")