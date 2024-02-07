import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tdt
import csv

df = pd.read_csv('D:/Chronic_Array/219/219-230929/PCA_Base.csv')
base_record = np.array(pd.read_csv('D:/Chronic_Array/Close_Loop/matrix_bin.csv',skiprows=0))
# print(df)

# read csv and extract array of base change:
coeffs = df.values

# 3x32 array of random number for testing
# transfer_matrix = np.random.rand(3, 32)

# Verify the shape
# if transfer_matrix.shape != (3, len(transfer_matrix[0])):
#     raise ValueError("Data shape does not match the expected 3x32 shape.")


#convert in the PC base


PC = np.dot(coeffs, base_record.T)





# plt.show()

# print(PC[1])
# Extract the PC2 Data
PC2 = coeffs[1]
PC3 = coeffs[2]
# print(len(PC3))

# rolling median of baseline PC2


PC2_buffer = np.empty(15)
PC2_buffer[:] = np.nan
# print(PC2_buffer)
PC2_basemed = []
for i in PC[1]:
    # print(PC2_buffer)
    PC2_buffer = PC2_buffer[1:]
    PC2_buffer = np.append(PC2_buffer,i)
    # print(PC2_buffer)
    PC2_basemed.append(np.nanmean(PC2_buffer))
    # PC2_basemed.append(np.nanmedian(PC2_buffer))

plt.plot(np.linspace(0,1, num=len(PC[1]) ), PC[1])
plt.plot(np.linspace(0,1, num=len(PC2_basemed) ), PC2_basemed)
plt.show()
med_mean =[]
PC2_buffer = np.empty(25)
PC2_buffer[:] = np.nan
for i in PC2_basemed:
    PC2_buffer = PC2_buffer[1:]
    PC2_buffer = np.append(PC2_buffer,i)
    med_mean.append(np.nanmean(PC2_buffer))




# print(PC2_basemed)
plt.hist(med_mean, bins='auto', alpha = 0.7)
plt.xlabel("med of Mean over 25 val")
plt.ylabel("counts")

plt.show()
