import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tdt
import csv

df = pd.read_csv('D:/Chronic_Array/219/219-230929/PCA_Base.csv')
base_record = np.array(pd.read_csv('D:/Chronic_Array/Close_Loop/matrix_bin.csv',skiprows=0))
max_base = np.max(base_record)
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
print(PC.shape)
# Extract the PC2 Data
PC2 = coeffs[1]
PC3 = coeffs[2]
# print(len(PC3))

# rolling median of baseline PC2
PC2_buffer = np.empty((1, 25))


# Compute 95th percentile
percentile_95 = np.percentile(PC[1], 99)
print(f"The 95th percentile value is: {percentile_95}")


RZ_IP = '10.1.0.100'

udp = tdt.TDTUDP(host = RZ_IP, send_type=np.float32, sort_codes=2, bits_per_bin=4)
syn = tdt.SynapseAPI()
currTime = 0
totalTime = 60 # sec


# experiment = syn.getKnownExperiments()
# print(experiment)

syn.setCurrentExperiment('CloseLoop_Stimulation')

# subject = syn.getKnownSubjects()
# print(subject)

syn.setCurrentSubject('219')


syn.setCurrentTank('D:/Chronic_Array/CLSTank')


# exp_name = syn.get

state = syn.getMode()

# for preview: 
if state < 1: syn.setMode(2)

# for recording: 
# if syn.getMode() < 1: syn.setMode(3)

"""Approach 1"""

# fr_export = []
# e = 1e-10
# count = 0
# res_bin = []
# buffer1 = np.zeros(25)

# while currTime < totalTime :
#     currTime = syn.getSystemStatus()['recordSecs']
#     data = udp.recv()
#     sort_code = 1
#     sc = data[sort_code-1]
#     fr_matrix = []
#     for ch in (1,2,6,10,17,22,25,27,28,29,30,31,32):
#         fr_matrix.append(sc[ch-1])
#     # Normalize values
#     fr_matrix = fr_matrix / max_base
#     # Multiply the two matrices to change basis into PC
#     result_matrix = np.dot(baseline, fr_matrix)
#     buffer1.pop(0)
#     buffer1.append(result_matrix)
#     PC2_proj = buffer1[12]
#     count += 1
#     # Check if any value in the matrix exceeds the 95th percentile in PC2
#     if np.any(PC2_proj > percentile_95) and count == 25:
#         udp.send(fr_matrix)
        




"""Approach 2"""

fr_export = []
e = 1e-10
stimTime = -5
R219 = (1,3,5,9,11,12,13,14,16,17,18,19,23,24,26,28,29,30,31,32)
while currTime < totalTime :
    currTime = syn.getSystemStatus()['recordSecs']
    data = udp.recv()
    sort_code = 1
    sc = data[sort_code-1]
    fr_matrix = []
    for ch in R219:
        fr_matrix.append(sc[ch-1])
    # Normalize values
    fr_matrix = (fr_matrix - np.min(fr_matrix)) / (np.max(fr_matrix) - np.min(fr_matrix) + e)
    # Multiply the two matrices to change basis into PC
    result_matrix = np.dot(PC2, fr_matrix.T)
    # Check if any value in the matrix exceeds the 95th percentile in PC2
    if result_matrix > percentile_95 and currTime - stimTime >= 5:
        # udp.send(fr_matrix)
        stimTime = currTime
        print(stimTime)
        print(result_matrix)


syn.setMode(0)        

# filename = "matrix_bin_proj.csv"
# header = ["ch_" + str(i) for i in range(len(data[0]))]
# with open(filename, 'w', newline="") as csvfile:
#     writer = csv.writer(csvfile)

#     writer.writerow(header)
#     for row in fr_export:
#         writer.writerow(row)

# # Extract the PC2 Data
# PC2 = data[1]

# # Plot histogram
# plt.hist(PC2, bins=10, color='blue', alpha=0.7, edgecolor='black')
# plt.title("Histogram of the second row")
# plt.xlabel("Projection")
# plt.ylabel("Frequency")
# # plt.grid(True)
# plt.show()

