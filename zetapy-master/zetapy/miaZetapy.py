import scipy.io
from zetapy import ifr, zetatest, zetatstest, zetatest2, zetatstest2, plotzeta, plottszeta, plotzeta2, plottszeta2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import convolve
import time
import csv
import matplotlib.pyplot as plt

# %% load and prepare some example data
# load data for example cell
with open("C:/Users/Maxime/Documents/GitHub/Data_Analysis/zetapy-master/zetapy/testNeuronZeta.csv", 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

time_before = 1
time_after = 2
t_tot = time_before + time_after
time_stim = 0.5
# retrieve the spike times as an array 
vecSpikeTimes1 = []
vecStimulusStartTimes = []
vecStimulusStopTimes = []
u = 0
for val in data:

    t_add = (t_tot*(u))
    u += 1
    for i in val:
        vecSpikeTimes1.append(float(i) + time_before + t_add)
    vecStimulusStartTimes.append(float(time_before + t_add))
    vecStimulusStopTimes.append(float(time_before + t_add + time_stim))
    
vecSpikeTimes1 = np.array(vecSpikeTimes1)
vecStimulusStartTimes= np.array(vecStimulusStartTimes)
vecStimulusStopTimes= np.array(vecStimulusStopTimes)


# x = range(len(vecSpikeTimes1))
# plt.plot(x,vecSpikeTimes1)
# plt.show()

# calculate instantaneous firing rate without performing the ZETA-test
# if we simply want to plot the neuron's response, we can use:
vecTime, vecRate, dIFR = ifr(vecSpikeTimes1, vecStimulusStartTimes)

# plot results
f, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(vecTime, vecRate)
ax.set(xlabel='Time after event (s)', ylabel='Instantaneous spiking rate (Hz)')
ax.set(title="A simple plot of the neuron's rate using ifr()")


# %% run the ZETA-test with specified parameters
# set random seed
np.random.seed(1)

# use minimum of trial-to-trial durations as analysis window size
dblUseMaxDur = np.min(np.diff(vecStimulusStartTimes))

# 50 random resamplings should give us a good enough idea if this cell is responsive.
# If the p-value is close to 0.05, we should increase this number.
intResampNum = 50

# what size of jittering do we want? (multiple of dblUseMaxDur; default is 2.0)
dblJitterSize = 2.0

# Do we want to plot the results?
boolPlot = True

# do we want to restrict the peak detection to for example the time during stimulus?
# Then put (0 1) here.
tplRestrictRange = (0, np.inf)

# do we want to compute the instantaneous firing rate?
boolReturnRate = True

# create a T by 2 array with stimulus onsets and offsets so we can also compute the t-test
arrEventTimes = np.transpose(np.array([vecStimulusStartTimes, vecStimulusStopTimes]))

# then run ZETA with those parameters
t = time.time()
dblZetaP, dZETA, dRate = zetatest(vecSpikeTimes1, arrEventTimes,
                                                dblUseMaxDur=dblUseMaxDur,
                                                intResampNum=intResampNum,
                                                dblJitterSize=dblJitterSize,
                                                boolPlot=boolPlot,
                                                tplRestrictRange=tplRestrictRange,
                                                boolReturnRate=boolReturnRate)

dblElapsedT2 = time.time() - t
print(f"\nSpecified parameters (elapsed time: {dblElapsedT2:.2f} s): \
      \nzeta-test p-value: {dblZetaP}\nt-test p-value:{dZETA['dblMeanP']}")


# Note on the latencies: while the peaks of ZETA and -ZETA can be useful for diagnostic purposes,
# they are difficult to interpret, so we suggest sticking to the peak time (vecLatencies[2]),
# which is more easily interpretable. Please read the paper for more information.
vecLatencies = dZETA['vecLatencies']

# %% run the time-series zeta-test
# take subselection of data
intUseTrialNum = 480
vecStimulusStartTimesTs = vecStimulusStartTimes[0:intUseTrialNum]
vecStimulusStopTimesTs = vecStimulusStopTimes[0:intUseTrialNum]
arrEventTimesTs = np.transpose(np.array([vecStimulusStartTimesTs, vecStimulusStopTimesTs]))

# first transform the data to time-series
print('\nRunning time-series zeta-test; This will take around 40 seconds\n')
dblStartT = 0
dblEndT = vecStimulusStopTimesTs[-1] + dblUseMaxDur*5
dblSamplingRate =  50.0   # simulate acquisition rate
dblSampleDur = 1/dblSamplingRate
vecTimestamps = np.arange(dblStartT, dblEndT+dblSampleDur, dblSampleDur)
vecSpikesBinned = np.histogram(vecSpikeTimes1, bins=vecTimestamps)[0]
vecTimestamps = vecTimestamps[0:-1]
dblSmoothSd = 1.0
intSmoothRange = 2*np.ceil(dblSmoothSd).astype(int)
vecFilt = norm.pdf(range(-intSmoothRange, intSmoothRange+1), 0, dblSmoothSd)
vecFilt = vecFilt / sum(vecFilt)

# pad array
intPadSize = np.floor(len(vecFilt)/2).astype(int)
vecData1 = np.pad(vecSpikesBinned, ((intPadSize, intPadSize)), 'edge')

# filter
vecData1 = convolve(vecData1, vecFilt, 'valid')

# set random seed
np.random.seed(1)

# time-series zeta-test with default parameters
t = time.time()
dblTsZetaP = zetatstest(vecTimestamps, vecData1, vecStimulusStartTimesTs)[0]
dblElapsedT3 = time.time() - t
print(f"\nDefault parameters (elapsed time: {dblElapsedT3:.2f} s):\ntime-series zeta-test p-value: {dblTsZetaP}\n")
# %% run time-series zeta-test with specified parameters
# set random seed
np.random.seed(1)
t = time.time()

# run test
print('\nRunning time-series zeta-test with specified parameters; This will take around 40 seconds\n')
dblTsZetaP2, dZetaTs = zetatstest(vecTimestamps, vecData1, arrEventTimesTs,
                                  dblUseMaxDur=None, intResampNum=100, boolPlot=True,
                                  dblJitterSize=2.0, boolDirectQuantile=False)

dblElapsedT4 = time.time() - t
print(f"\nSpecified parameters (elapsed time: {dblElapsedT4:.2f} s): \
      \ntime-series zeta-test p-value: {dblTsZetaP2}\nt-test p-value:{dZetaTs['dblMeanP']}")
