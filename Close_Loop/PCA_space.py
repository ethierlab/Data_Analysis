import tdt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data = np.array(pd.read_csv('D:/Chronic_Array/Close_Loop/matrix_bin.csv',skiprows=0))
print(data.shape)
# Raw_data = data.transpose()
# print(Raw_data.shape)

fr = []
for channel in data:
  fr.append(channel)




scalar = MinMaxScaler(feature_range=(0,1))
X = scalar.fit_transform(fr)
# pca.fit(fr)
pca = PCA(n_components=3)
pca.fit(X)
coeffs = pca.components_
print(coeffs)

# save to a csv
df = pd.DataFrame(coeffs)
df.to_csv('D:/Chronic_Array/219/219-230929/PCA_Base.csv', index=False)
