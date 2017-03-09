import numpy as np
import pandas as pd
X = np.genfromtxt("digit.txt")
df = pd.DataFrame(X)
df.to_pickle("digits.pkl")