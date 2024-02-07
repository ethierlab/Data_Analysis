import tdt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv



RZ_IP = '10.1.0.100'

udp = tdt.TDTUDP(host = RZ_IP, send_type=np.float32, sort_codes=2, bits_per_bin=4)
syn = tdt.SynapseAPI()
currTime = 0
totalTime = 300 # sec

# experiment = syn.getKnownExperiments()
# print(experiment)

syn.setCurrentExperiment('CloseLoop_Stimulation')

# subject = syn.getKnownSubjects()
# print(subject)

syn.setCurrentSubject('219')


syn.setCurrentTank('D:/Chronic_Array/CLSTank')


# exp_name = syn.get

state = syn.getMode()
R219 = (1,3,5,9,11,12,13,14,16,17,18,19,23,24,26,28,29,30,31,32)
R220 = (1,2,6,10,17,22,25,27,28,29,30,31,32)
# for preview: 
if state < 1: syn.setMode(3)

# for recording: 
# if syn.getMode() < 1: syn.setMode(3)

fr_export = []

while currTime < totalTime :
    currTime = syn.getSystemStatus()['recordSecs']
    data = udp.recv()
    sort_code = 1
    sc = data[sort_code-1]
    fr = []
    for ch in R219:
    # for ch in range(1, 33):
    # channel = 10
        fr.append(sc[ch-1])    
        #append to the fr by channel by time bins:
    fr_export.append(fr)


syn.setMode(0)
filename = "matrix_bin.csv"
header = ["ch_" + str(i) for i in R219]
# header = ["ch_" + str(i) for i in range(1,33)]
with open(filename, 'w', newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for row in fr_export:
        writer.writerow(row)