import adi
import numpy as np
import matplotlib.pyplot as plt
import data_analysis as da # Vincent's function
from scipy.signal import find_peaks
from scipy.signal import medfilt 
from scipy.stats import linregress
from scipy.signal import butter
from scipy.signal import sosfilt
from scipy.optimize import curve_fit
from scipy import integrate
from sympy import Eq, symbols, solve
from sympy import solve, symbols, Eq, exp, N, GreaterThan
import numpy as np
import os
import math
import tkinter as tk
from tkinter import filedialog
import statistics as st

class Psth:
    "For psth production and visualisation"
    def __init__(self, autoPath = "", typeStim = "opto"):
        "If autoPath empty, path is choosen in dialog. Default type stim is opto"
        self.saveFigPathFolder = "C:/Users/Maxime/Desktop/dossier"
        self.c_data = {} # dictionnaire compile les données par channel et bloc
        self.freq = 10000 # frequence d'échantillonnage
        self.t_inf = 0.05
        self.t_supp = 0.05
        self.time_window = 0.0
        self.select = 0
        self.experiment = 0
        self.typeStim = typeStim
        self.adi_out = []
        self.courant_val = [] # valeur des courant opto
        self.pulseWidth = [] # valeur des pulses pour la courbe de recrutement
        self.frequence =[] # valeur des fréquences pour la courbe de recrutement
        self.psth_compil = []
        self.min_max = []
        self.typeStim = typeStim
        self.psth_time = []
        self.signal_channel_gain = 1
        self.borneTemp = [[0, 0.0075], [0.0075, 0.015], [0, 0.025]]
        self.borneTemp = [[0.0015, 0.025]]
        self.peak2peak_amp = []
        self.statsEmg = {}
        self.statsForce = {}
        self.calibrationSenseur = {} # regression lin result : voltage by gram, keys = slope, intercept,r2
        self.resumeDataLatence = {} # [latence, AUC, val_courant]
        self.paraSigmoid =[] # paramètres de la sigmoide L,b,x0,k

        if (len(autoPath)<1):
            root = tk.Tk()
            root.withdraw()
            self.path = filedialog.askdirectory()
        else:
            self.path = autoPath
          
    def loadDataFromDir(self, preFileNameChara = "_", postFileNameChara = "ma"):
        """ Read the 'adi' file in the folder. 1 file, 1 recording, 1 condition, 1 signal channel and 1 event channel(stim)
            Extract the stimulation curent amplitude value in the file name, base on pre and post File character. If only one file
            in folder : 
        """
            ## # All id numbering is 1 based, first channel, first block
            # When indexing in Python we need to shift by 1 for 0 based indexing
            # Functions however respect the 1 based notation ...
            
        if not os.path.exists(self.path):
            print(f"Error: The directory '{self.path}' does not exist.")
            return
        # List all the files in the given directory
        files = os.listdir(self.path)
        n_files = len(files)
        
        
        # condition variable indépendante : intensité ma
        courant_val = []
        adi_out = []

        if (postFileNameChara == "ma"):
            match n_files:
                case 0:
                    print("Error no files to analyse")
                case 1:
                    if files[0].endswith(".adicht"):
                        txt_path = os.path.join(self.path, files[0])
                        ind1 = files[0].rfind(preFileNameChara)
                        ind2 = files[0].rindex(postFileNameChara)
                        courant_val.append(float(files[0][ind1+1:ind2]))
                        f = adi.read_file(txt_path)
                        adi_out.append(f)
                    else:
                        print("Error no labchart files to analyse")

                case _:
                    for file_name in files:
                        if file_name.endswith(".adicht"):
                            txt_path = os.path.join(self.path, file_name)
                            ind1 = file_name.rfind(preFileNameChara)
                            ind2 = file_name.rindex(postFileNameChara)
                            courant_val.append(float(file_name[ind1+1:ind2]))
                            f = adi.read_file(txt_path)
                            adi_out.append(f)
                        else:
                            print("Error at least one file is not a labchart one")
                    
            
            self.adi_out = adi_out
            self.courant_val = courant_val
        elif (postFileNameChara == "ms"):
            # condition variable indépendante : intensité ms
            pulse = []
            adi_out = []

            match n_files:
                case 0:
                    print("Error no files to analyse")
                case 1:
                    if files[0].endswith(".adicht"):
                        txt_path = os.path.join(self.path, files[0])
                        ind1 = files[0].rfind(preFileNameChara)
                        ind2 = files[0].rindex(postFileNameChara)
                        pulse.append(float(files[0][ind1+1:ind2]))
                        f = adi.read_file(txt_path)
                        adi_out.append(f)
                    else:
                        print("Error no labchart files to analyse")

                case _:
                    for file_name in files:
                        if file_name.endswith(".adicht"):
                            txt_path = os.path.join(self.path, file_name)
                            ind1 = file_name.rfind(preFileNameChara)
                            ind2 = file_name.rindex(postFileNameChara)
                            pulse.append(float(file_name[ind1+1:ind2]))
                            f = adi.read_file(txt_path)
                            adi_out.append(f)
                        else:
                            print("Error at least one file is not a labchart one")
                    
            self.adi_out = adi_out
            self.pulseWidth = pulse
        elif (postFileNameChara == "hz"):
            # condition variable indépendante : intensité ms
            frequence = []
            adi_out = []

            match n_files:
                case 0:
                    print("Error no files to analyse")
                case 1:
                    if files[0].endswith(".adicht"):
                        txt_path = os.path.join(self.path, files[0])
                        ind1 = files[0].rfind(preFileNameChara)
                        ind2 = files[0].rindex(postFileNameChara)
                        frequence.append(float(files[0][ind1+1:ind2]))
                        f = adi.read_file(txt_path)
                        adi_out.append(f)
                    else:
                        print("Error no labchart files to analyse")

                case _:
                    for file_name in files:
                        if file_name.endswith(".adicht"):
                            txt_path = os.path.join(self.path, file_name)
                            ind1 = file_name.rfind(preFileNameChara)
                            ind2 = file_name.rindex(postFileNameChara)
                            frequence.append(float(file_name[ind1+1:ind2]))
                            f = adi.read_file(txt_path)
                            adi_out.append(f)
                        else:
                            print("Error at least one file is not a labchart one")
                    
            self.adi_out = adi_out
            self.frequence = frequence
   
    def loadLabchartFromDir(self):
        """ Read the 'adi' file in the folder. Only one labchart file in the directory for calibration or paramètre optimaux.
        Pas fait pour experience 1
        """
            ## # All id numbering is 1 based, first channel, first block
            # When indexing in Python we need to shift by 1 for 0 based indexing
            # Functions however respect the 1 based notation ...
            
        if not os.path.exists(self.path):
            print(f"Error: The directory '{self.path}' does not exist.")
            return
        # List all the files in the given directory
        files = os.listdir(self.path)
        n_files = len(files)
        adi_out = []
        
        match n_files:
            case 0: # if no file
                print("Error no files to analyse")
            case 1: # one file available
                if files[0].endswith(".adicht"):
                    txt_path = os.path.join(self.path, files[0])
                    
                    f = adi.read_file(txt_path)
                    adi_out.append(f) # Output of adi read
                    
                else:
                    print("Error no labchart files to analyse")
        for i in range(len(adi_out)):
            record = adi_out[i]
            for channel in range(record.n_channels):
                for bloc in range(record.n_records):
                    self.c_data[(channel,bloc)] =  record.channels[channel].get_data(bloc+1)
                    
            
        self.adi_out = adi_out
        
    def fromChannel2Psth(self, t_inf, t_supp, numberSignal, numberEvent, OnePulsePerEvent = True):
        """t_inf, t_supp : time in second before and after the event
            numberSignal : channel number, numberEvent : event channel number
            OnePulsePerEvent : True = each event is not a train. False = each event is a train, first spike took as onset
           
        """

        # Si le canal doit etre l'emg rectifié ou la force

        min_max=[0,0]
        psth_compil = []
        
        for i in range(len(self.adi_out)):
            record = self.adi_out[i]
            for channel in range(record.n_channels):
                for bloc in range(record.n_records):
                    self.c_data[(channel,bloc)] =  record.channels[channel].get_data(bloc+1)
                    
            signal_channel=self.c_data[(numberSignal - 1, 0)] # Value associate to the channel bloc
            event_channel=self.c_data[(numberEvent - 1, 0)] # Value associate to the channel bloc
            signal_channel = signal_channel / self.signal_channel_gain
            time = np.arange(len(signal_channel))
            time  = time/self.freq
            index = da.find_event_index(event_channel, self.freq, self.select, self.time_window, self.experiment)
            if not OnePulsePerEvent:
                index = da.takeFirstPeak(index, .2, .5, self.freq) # second argument = min inter train and 3e = max intra train
            sample = da.cut_individual_event(t_inf, t_supp, index, signal_channel, self.freq)
            
            min_psth, moy_psth, max_psth = da.PSTH(sample)
            mini = min(moy_psth)
            maxi = max(moy_psth)
            min_max = [min([min_max[0], mini]), max([min_max[1], maxi])]
            if (len(self.adi_out)>1):
                psth_compil.append(moy_psth)
            else:
                psth_compil = [moy_psth]
        self.t_inf = t_inf
        self.t_supp = t_supp
        self.psth_compil = psth_compil
        self.min_max = min_max
    
    def fromChannel2PsthRectEmg(self, t_inf, t_supp, numberSignal, numberEvent, OnePulsePerEvent = True):
        """t_inf, t_supp : time in second before and after the event
            numberSignal : channel number, numberEvent : event channel number
            OnePulsePerEvent : True = each event is not a train. False = each event is a train, first spike took as onset
           
        """

        min_max=[0,0]
        psth_compil = []
        
        for i in range(len(self.adi_out)):
            record = self.adi_out[i]
            for channel in range(record.n_channels):
                for bloc in range(record.n_records):
                    self.c_data[(channel,bloc)] =  record.channels[channel].get_data(bloc+1)
                    
            signal_channel=self.c_data[(numberSignal - 1, 0)] # Value associate to the channel bloc
            
            event_channel=self.c_data[(numberEvent - 1, 0)] # Value associate to the channel bloc
            signal_channel = signal_channel / self.signal_channel_gain
            
            time = np.arange(len(signal_channel))
            time  = time/self.freq
            index = da.find_event_index(event_channel, self.freq, self.select, self.time_window, self.experiment)
            if not OnePulsePerEvent:
                index = da.takeFirstPeak(index, .2, .5, self.freq)
            
            # Rectification du signal
            self.rectEmg(signal_channel, index, 2, showPlot = False, emg_max_latency = 0.015, thresholdMinEmg = 0.1)
            sample = da.cut_individual_event(t_inf, t_supp, index, self.statsEmg["RectSignal"], self.freq)
            min_psth, moy_psth, max_psth = da.PSTH(sample)
            mini = min(moy_psth)
            maxi = max(moy_psth)
            min_max = [min([min_max[0], mini]), max([min_max[1], maxi])]
            if (len(self.adi_out)>1):
                psth_compil.append(moy_psth)
            else:
                psth_compil = [moy_psth]
        self.t_inf = t_inf
        self.t_supp = t_supp
        self.psth_compil = psth_compil
        self.min_max = min_max
    
    def fromChannel2PsthForce(self, t_inf, t_supp, numberSignal, numberEvent, OnePulsePerEvent = True):
        """t_inf, t_supp : time in second before and after the event
            numberSignal : channel number, numberEvent : event channel number
            OnePulsePerEvent : True = each event is not a train. False = each event is a train, first spike took as onset
           
        """

        min_max=[0,0]
        psth_compil = []
        
        for i in range(len(self.adi_out)):
            record = self.adi_out[i]
            for channel in range(record.n_channels):
                for bloc in range(record.n_records):
                    self.c_data[(channel,bloc)] =  record.channels[channel].get_data(bloc+1)
                    
            signal_channel=self.c_data[(numberSignal - 1, 0)] # Value associate to the channel bloc
            
            event_channel=self.c_data[(numberEvent - 1, 0)] # Value associate to the channel bloc
            signal_channel = signal_channel / self.signal_channel_gain
            
            time = np.arange(len(signal_channel))
            time  = time/self.freq
            index = da.find_event_index(event_channel, self.freq, self.select, self.time_window, self.experiment)
            if not OnePulsePerEvent:
                index = da.takeFirstPeak(index, .2, .5, self.freq)
            
            # Aire sous la courbe
            self.forceAreaUnderCurve(signal_channel, index, 2, showPlot = False)
            sample = da.cut_individual_event(t_inf, t_supp, index, self.statsForce["forceCalib"], self.freq)
            
            min_psth, moy_psth, max_psth = da.PSTH(sample)
            mini = min(moy_psth)
            maxi = max(moy_psth)
            min_max = [min([min_max[0], mini]), max([min_max[1], maxi])]
            if (len(self.adi_out)>1):
                psth_compil.append(moy_psth)
            else:
                psth_compil = [moy_psth]
        self.t_inf = t_inf
        self.t_supp = t_supp
        self.psth_compil = psth_compil
        self.min_max = min_max
    
    def latenceVsEmg(self, OnePulsePerEvent, showPlotLatence, showPlot):
        """calcul la latence en fonction de l'emg rectifié. Part du fichier importation mis sous la forme dictionnaire c_data
        """
        
        
        for fichier in range(len(self.adi_out)): # Loop fichier par fichier : 1 fichier = 1 intensité dans ce cas ci
            val_courant = self.courant_val[fichier] # établi dans la fonction "loadDataFromDir"
            record = self.adi_out[fichier]
            for channel in range(record.n_channels):
                for bloc in range(record.n_records): # un seul bloc pour l'expérience 1
                    self.c_data[(channel,bloc)] =  record.channels[channel].get_data(bloc+1)
                    
            signal_channel=self.c_data[(0, 0)] # Value associate to the channel bloc
            event_channel=self.c_data[(1, 0)] # Value associate to the channel bloc
            signal_channel = signal_channel / self.signal_channel_gain
            time = np.arange(len(signal_channel))
            time  = time/self.freq
            index = da.find_event_index(event_channel, self.freq, self.select, self.time_window, self.experiment)
            if not OnePulsePerEvent:
                index = da.takeFirstPeak(index, .2, .5, self.freq)
            
            # Rectification du signal
            sos = butter(2, 50, 'hp', fs=self.freq, output='sos')
            signalFilt = sosfilt(sos, signal_channel) # high pass filter 50hz
            rectSignalFilt = abs(signalFilt)
            sos = butter(2, 10, 'lp', fs=self.freq, output='sos')
            signalFiltFinal = sosfilt(sos, rectSignalFilt) # high pass filter 50hz
            # Area under curve
            emg_max_latency = 0.015
            thresholdMinEmg = 0.01
            dx_time = 1/self.freq
            
            # peak to peak and latency
            window_end_bin = math.floor(self.freq * emg_max_latency)
            latency = []
            areaUnderCurveCompil = [] # par pulse 
            val_stimulation = [] # repetition de la valeur du courant de stimulation
            data = [] # tableau de ligne : [latency, areaUnderCurve, val_stimulation]
            subplot_mxn =  self.findMxNsubplotGrid(len(index))
            u=0
            for i in index:
                seg = signal_channel[i:(i + window_end_bin)]
                segSum = sum(seg * dx_time)*window_end_bin
                
                if (max(seg) > thresholdMinEmg): # Calcul les latences uniquement si il y a emg
                    ind = math.floor(self.find_latency(seg, window_size=10, k=0.1))
                    latency.append(ind / self.freq)
                    areaUnderCurveCompil.append(segSum)
                    val_stimulation.append(val_courant)
                    data.append([ind/self.freq, segSum, val_courant]) # [latence, AUC, val_courant]

                    if showPlotLatence:
                        u=u+1
                        plt.subplot(subplot_mxn[0], subplot_mxn[1],u)
                        x = np.arange(0,len(seg))/self.freq
                        plt.plot(x, seg, "-k", label="Intensité = "+ str(val_courant))
                        plt.legend(fontsize=5, loc=4)
                        plt.plot(x[ind], seg[ind],"or")
                        plt.xlim([0,.02])
                        plt.xlabel("time (s)")
                        plt.ylabel("EMG (V)")
                        
                        
            if (len(latency)>1):
                latency_ave = sum(latency) / len(latency)
                latency_std = st.stdev(latency)
                latency_min = min(latency)
                latency_max = max(latency)
                AUC_ave = sum(areaUnderCurveCompil)/len(areaUnderCurveCompil)
                AUC_std = st.stdev(latency)

            else : 
                latency_ave = 0
                latency_std = 0
                latency_min = 0
                latency_max = 0
                AUC_ave = 0
                AUC_std = 0
                data = [0, 0, 0]
            if showPlotLatence:# plot les latences
                plt.show()
             
            # résumé données :
            self.resumeDataLatence.update({(fichier,"data"): data }) # [latence, AUC, val_courant]
            self.resumeDataLatence.update({(fichier,"statLatence"):[latency_ave, latency_std, latency_min, latency_max]})
            self.resumeDataLatence.update({(fichier,"statEMG"):[AUC_ave,AUC_std]})
            
        
        if showPlot:
            x = []
            y = []
            c = []
            for i in range(len(self.adi_out)):
                x.append(self.resumeDataLatence[(i,"statEMG")][0])
                y.append(self.resumeDataLatence[(i,"statLatence")][0])
                c.append(self.resumeDataLatence[(i,"statEMG")][1])
            plt.scatter(x, y)
            plt.errorbar(x, y, yerr = c, fmt="o")
            plt.title("latence de l'EMG selon son amplitude")
            plt.show()
            
    def showAllPsth(self, saveplotName = ""):
        "Plot psth of each stimulation on the same page. Make a diff"
        subplot_mxn =  self.findMxNsubplotGrid(len(self.adi_out))
        fin = len(self.adi_out)
        for i in range(fin):
            expression = "plt.subplot(" + str(subplot_mxn[0]) +","+ str(subplot_mxn[1])+"," + str(i+1) + ")"
            exec(expression)
            
            if (self.typeStim == "elect"):
            # ordre des graphique en courant ascendant
                ind = np.argsort(self.courant_val)
                puissance = self.calibCourantPuissance(self.courant_val, self.typeStim)
                time = np.arange(len(self.psth_compil[ind[i]]))
                time = (time/self.freq)-self.t_inf
                plt.plot(time,self.psth_compil[ind[i]], label= str(puissance[ind[i]]) + " uA")
                plt.xlabel("time (s)", fontsize=4)
                plt.ylabel("EMG (V)", fontsize=4)
                plt.legend(fontsize=4)
                plt.ylim((self.min_max[0],self.min_max[1]))
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
            if (self.typeStim == "opto"):
            # ordre des graphique en courant ascendant
                ind = np.argsort(self.courant_val)
                puissance = self.calibCourantPuissance(self.courant_val, self.typeStim)
                time = np.arange(len(self.psth_compil[ind[i]]))
                time = (time/self.freq)-self.t_inf
                plt.plot(time,self.psth_compil[ind[i]], label= str(self.courant_val[ind[i]]) + " mW")
                plt.xlabel("time (s)", fontsize=4)
                plt.ylabel("EMG (V)", fontsize=4)
                plt.legend(fontsize=4)
                plt.ylim((self.min_max[0],self.min_max[1]))
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
        if (len(saveplotName) >= 1):
            self.saveFigure(saveplotName)
        plt.show()
        
        self.psth_time = time

    def showAllPsthPulse(self, saveplotName = ""):
        "Plot psth of each stimulation on the same page. Make a diff"
        subplot_mxn =  self.findMxNsubplotGrid(len(self.adi_out))
        fin = len(self.adi_out)
        for i in range(fin):
            expression = "plt.subplot(" + str(subplot_mxn[0]) +","+ str(subplot_mxn[1])+"," + str(i+1) + ")"
            exec(expression)
            
            if (self.typeStim == "elect"):
            # ordre des graphique en courant ascendant
                ind = np.argsort(self.pulseWidth)
                time = np.arange(len(self.psth_compil[ind[i]]))
                time = (time/self.freq)-self.t_inf
                plt.plot(time,self.psth_compil[ind[i]], label= str(self.pulseWidth[ind[i]]) + " ms")
                plt.xlabel("time (s)", fontsize=4)
                plt.ylabel("EMG (V)", fontsize=4)
                plt.legend(fontsize=4)
                plt.ylim((self.min_max[0],self.min_max[1]))
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
            if (self.typeStim == "opto"):
            # ordre des graphique en courant ascendant
                ind = np.argsort(self.pulseWidth)
                time = np.arange(len(self.psth_compil[ind[i]]))
                time = (time/self.freq)-self.t_inf
                plt.plot(time,self.psth_compil[ind[i]], label= str(self.pulseWidth[ind[i]]) + " ms")
                plt.xlabel("time (s)", fontsize=4)
                plt.ylabel("EMG (V)", fontsize=4)
                plt.legend(fontsize=4)
                plt.ylim((self.min_max[0],self.min_max[1]))
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
        if (len(saveplotName) >= 1):
            self.saveFigure(saveplotName)
        plt.show()
        
        self.psth_time = time

    def showAllPsthFrequence(self, saveplotName = ""):
        "Plot psth of each stimulation on the same page. Make a diff"
        subplot_mxn =  self.findMxNsubplotGrid(len(self.adi_out))
        fin = len(self.adi_out)
        for i in range(fin):
            expression = "plt.subplot(" + str(subplot_mxn[0]) +","+ str(subplot_mxn[1])+"," + str(i+1) + ")"
            exec(expression)
            
            if (self.typeStim == "elect"):
            # ordre des graphique en courant ascendant
                ind = np.argsort(self.frequence)
                time = np.arange(len(self.psth_compil[ind[i]]))
                time = (time/self.freq)-self.t_inf
                plt.plot(time,self.psth_compil[ind[i]], label= str(self.frequence[ind[i]]) + " hz")
                plt.xlabel("time (s)", fontsize=4)
                plt.ylabel("EMG (V)", fontsize=4)
                plt.legend(fontsize=4)
                plt.ylim((self.min_max[0],self.min_max[1]))
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
            if (self.typeStim == "opto"):
            # ordre des graphique en courant ascendant
                ind = np.argsort(self.frequence)
                time = np.arange(len(self.psth_compil[ind[i]]))
                time = (time/self.freq)-self.t_inf
                plt.plot(time,self.psth_compil[ind[i]], label= str(self.frequence[ind[i]]) + " hz")
                plt.xlabel("time (s)", fontsize=4)
                plt.ylabel("EMG (V)", fontsize=4)
                plt.legend(fontsize=4)
                plt.ylim((self.min_max[0],self.min_max[1]))
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
        if (len(saveplotName) >= 1):
            self.saveFigure(saveplotName)
        plt.show()
        
        self.psth_time = time

    def calibCourantPuissance(self,x,type:str):
        "Calibration en puissance pour la stimulation optogénétique et en courrant pour l'électrique"
        if (type == "opto"):
            m = 0.8473 # slope
            b = -9.6721 # ord à l'abscisse
            x=np.array(x)
            y=np.array([])
            y = m*x + b
            y = np.floor(y)
            return y
        if (type == "elect"):
            return x

    def sigmoid(self, x, L ,x0, k, b):
            "Equation d'une sigmoid"
            y = L / (1 + np.exp(-k*(x-x0))) + b
            return y

    def sigmoidFit(self, xdata, ydata):
        ""
        p0 = [max(ydata), np.median(xdata),1,min(ydata)] # les paramètres initiaux.[L, x0, k, b] : L le max de la courbe, b : offset en y, k scaling input, x0 :
        # la moitié du output
        popt, pcov = curve_fit(self.sigmoid, xdata, ydata, p0, method='lm')
        self.paraSigmoid = popt
        
    def findSigmoidYValue(self,pourcentageMax):

        "Valeux en x correspondant au pourcentage du maximum en y"
        
        l, x0, k, b, x, y, rep, cible = symbols('l x0 k b x y rep cible')

        # paramètres sigmoid
        l = self.paraSigmoid[0]
        x0 = self.paraSigmoid[1]
        k = self.paraSigmoid[2] 
        b = self.paraSigmoid[3]

       

        cible = (l+b)*pourcentageMax/100
        y = l / (1 + exp(-k*(x-x0))) + b
        equation = GreaterThan(y, cible)
        rep=solve(equation, x)
        print(rep)
        strRep = str(rep)
        if (strRep == "True"):
            valeur = 0
        elif (strRep == "False"):
            valeur = x0*2
        else :
            position = strRep.find('<')
            if (strRep[:position].find('x')== -1):
                valeur = round(float(strRep[:position]),2)
            else:
                valeur = round(float(strRep[position+3:]),2)

        return valeur
    
    def showSigmoidTableValue(self):
        "Show table of different values x and y based on sigmoid fit"
        cell_Text = [] # Le contenu de la table
        col_Labels = ["Stimulation Intensité", "Force maximale (%)"]
        forceMaxPourc = range(0,110,10)
        
        for val in forceMaxPourc:
            valeur = self.findSigmoidYValue(val)
            cell_Text.append([str(valeur), str(val)])
            print(cell_Text)
        
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis("off")
        ax.axis("tight")

        ax.table(cellText = cell_Text, colLabels=col_Labels, loc="top")
        fig.tight_layout()
        plt.show()
        
    def courbeRecrutement(self, borneTemp, titre:str, saveplotName = ""):
         # chan : psth array, borneTemp : borne à extraire peak amp
         # utilise le dernier psth conçu
        
        etendu = []
        auc = [] # aire sous la courbe
        signMax = [] # le maximum retrouvé
        ind = np.argsort(self.courant_val)
        s_courant_val = np.sort(self.courant_val)
        
        ind_t1 = borneTemp[0]
        ind_t2 = borneTemp[1]
        res = [idx for idx, val in enumerate(self.psth_time) if val > ind_t1]
        ind_t1=res[0]
        res = [idx for idx, val in enumerate(self.psth_time) if val < ind_t2]
        ind_t2=res[-1]

        for channel in self.psth_compil:
            signal_borne = channel[ind_t1 : ind_t2]
            min_signal = min(signal_borne)
            max_signal = max(signal_borne)
            etendu.append(max_signal - min_signal)
            # calcul aire sous la courbe
            # Area under curve
            dx_time = 1/self.freq
            aeraUnderCurve = sum(np.array(signal_borne) * dx_time)*(ind_t2 - ind_t1)
            auc.append(aeraUnderCurve) # aire sous la courbe compilée
            signMax.append(max_signal) # maximum compilé

        
        etendu = np.array(etendu)
        etendu = etendu[ind]
        
        auc = np.array(auc)
        auc = auc[ind]

        signMax = np.array(signMax)
        signMax = signMax[ind]

        # Mise en graphique

        # calcul de la sigmoide :
        
        self.sigmoidFit(s_courant_val, signMax)
        paraSigmoid = self.paraSigmoid
        x = np.linspace(0, max(s_courant_val), 100)
        fitY = self.sigmoid(x, paraSigmoid[0], paraSigmoid[1], paraSigmoid[2], paraSigmoid[3])
        # vv = range(0,100,10)
        # for v in vv:
        #     print(self.findSigmoidYValue(v), "v = ",v)
        # Graphique
        fig, ax = plt.subplots(2,sharex=True)
        ax[0].plot(s_courant_val, signMax,"-ob")
        ax[0].plot(x, fitY,"-k")
        ax[0].set(xlabel="Courant (mA)", ylabel = "Valeur Maximale")
        ax[0].set_title("Valeur Maximale" + titre)
        
        # calcul sigmoid
        self.sigmoidFit(s_courant_val, auc)
        paraSigmoid = self.paraSigmoid
        x = np.linspace(0, max(s_courant_val), 100)
        fitY = self.sigmoid(x, paraSigmoid[0], paraSigmoid[1], paraSigmoid[2], paraSigmoid[3])

        # Graphique
        ax[1].plot(s_courant_val, auc,"-ob")
        ax[1].plot(x, fitY,"-k")
        ax[1].set(xlabel = "Courant (mA)", ylabel = "AUC(Vs)")
        ax[1].set_title("Aire sous courbe" + titre)


        
        
        if (len(saveplotName) >= 1):
            self.saveFigure(saveplotName)
        plt.show()

    def courbeRecrutementPulse(self, borneTemp, titre:str, saveplotName = ""):
         # chan : psth array, borneTemp : borne à extraire peak amp
         # utilise le dernier psth conçu
        
        etendu = []
        auc = [] # aire sous la courbe
        signMax = [] # le maximum retrouvé
        ind = np.argsort(self.pulseWidth)
        s_pulseWidth = np.sort(self.pulseWidth)
        
        ind_t1 = borneTemp[0]
        ind_t2 = borneTemp[1]
        res = [idx for idx, val in enumerate(self.psth_time) if val > ind_t1]
        ind_t1=res[0]
        res = [idx for idx, val in enumerate(self.psth_time) if val < ind_t2]
        ind_t2=res[-1]

        for channel in self.psth_compil:
            signal_borne = channel[ind_t1 : ind_t2]
            min_signal = min(signal_borne)
            max_signal = max(signal_borne)
            etendu.append(max_signal - min_signal)
            # calcul aire sous la courbe
            # Area under curve
            dx_time = 1/self.freq
            aeraUnderCurve = sum(np.array(signal_borne) * dx_time)*(ind_t2 - ind_t1)
            auc.append(aeraUnderCurve) # aire sous la courbe compilée
            signMax.append(max_signal) # maximum compilé

        
        etendu = np.array(etendu)
        etendu = etendu[ind]
        
        auc = np.array(auc)
        auc = auc[ind]

        signMax = np.array(signMax)
        signMax = signMax[ind]

        # Mise en graphique

        # calcul de la sigmoide :
        
        self.sigmoidFit(s_pulseWidth, signMax)
        paraSigmoid = self.paraSigmoid
        x = np.linspace(0, max(s_pulseWidth), 100)
        fitY = self.sigmoid(x, paraSigmoid[0], paraSigmoid[1], paraSigmoid[2], paraSigmoid[3])
        
        # Graphique
        fig, ax = plt.subplots(2,sharex=True)
        ax[0].plot(s_pulseWidth, signMax,"-ob")
        ax[0].plot(x, fitY,"-k")
        ax[0].set(xlabel="Pulse width (ms)", ylabel = "Valeur Maximale")
        ax[0].set_title("Valeur Maximale" + titre)
        
        # calcul sigmoid
        self.sigmoidFit(s_pulseWidth, auc)
        paraSigmoid = self.paraSigmoid
        x = np.linspace(0, max(s_pulseWidth), 100)
        fitY = self.sigmoid(x, paraSigmoid[0], paraSigmoid[1], paraSigmoid[2], paraSigmoid[3])

        # Graphique
        ax[1].plot(s_pulseWidth, auc,"-ob")
        ax[1].plot(x, fitY,"-k")
        ax[1].set(xlabel = "Pulse width (ms)", ylabel = "AUC(Vs)")
        ax[1].set_title("Aire sous courbe" + titre)
        
        
        if (len(saveplotName) >= 1):
            self.saveFigure(saveplotName)
        plt.show()

    def courbeRecrutementFrequence(self, borneTemp, titre:str, saveplotName = ""):
         # chan : psth array, borneTemp : borne à extraire peak amp
         # utilise le dernier psth conçu
        
        etendu = []
        auc = [] # aire sous la courbe
        signMax = [] # le maximum retrouvé
        ind = np.argsort(self.frequence)
        s_frequence = np.sort(self.frequence)
        
        ind_t1 = borneTemp[0]
        ind_t2 = borneTemp[1]
        res = [idx for idx, val in enumerate(self.psth_time) if val > ind_t1]
        ind_t1=res[0]
        res = [idx for idx, val in enumerate(self.psth_time) if val < ind_t2]
        ind_t2=res[-1]

        for channel in self.psth_compil:
            signal_borne = channel[ind_t1 : ind_t2]
            min_signal = min(signal_borne)
            max_signal = max(signal_borne)
            etendu.append(max_signal - min_signal)
            # calcul aire sous la courbe
            # Area under curve
            dx_time = 1/self.freq
            aeraUnderCurve = sum(np.array(signal_borne) * dx_time)*(ind_t2 - ind_t1)
            auc.append(aeraUnderCurve) # aire sous la courbe compilée
            signMax.append(max_signal) # maximum compilé

        
        etendu = np.array(etendu)
        etendu = etendu[ind]
        
        auc = np.array(auc)
        auc = auc[ind]

        signMax = np.array(signMax)
        signMax = signMax[ind]

        # Mise en graphique

        # calcul de la sigmoide :
        
        self.sigmoidFit(s_frequence, signMax)
        paraSigmoid = self.paraSigmoid
        x = np.linspace(0, max(s_frequence), 100)
        fitY = self.sigmoid(x, paraSigmoid[0], paraSigmoid[1], paraSigmoid[2], paraSigmoid[3])
        
        # Graphique
        fig, ax = plt.subplots(2,sharex=True)
        ax[0].plot(s_frequence, signMax,"-ob")
        ax[0].plot(x, fitY,"-k")
        ax[0].set(xlabel="Frequence train (hz)", ylabel = "Valeur Maximale")
        ax[0].set_title("Valeur Maximale" + titre)
        
        # calcul sigmoid
        self.sigmoidFit(s_frequence, auc)
        paraSigmoid = self.paraSigmoid
        x = np.linspace(0, max(s_frequence), 100)
        fitY = self.sigmoid(x, paraSigmoid[0], paraSigmoid[1], paraSigmoid[2], paraSigmoid[3])

        # Graphique
        ax[1].plot(s_frequence, auc,"-ob")
        ax[1].plot(x, fitY,"-k")
        ax[1].set(xlabel = "Frequence train (hz)", ylabel = "AUC(Vs)")
        ax[1].set_title("Aire sous courbe" + titre)
        
        
        if (len(saveplotName) >= 1):
            self.saveFigure(saveplotName)
        plt.show()

    def peak2peak(self, saveplotName = ""):
        # chan : psth array, borneTemp : borne à extraire peak amp
        dict={}
        etendu = []
        ind = np.argsort(self.courant_val)
        s_courant_val = np.sort(self.courant_val)
        for borne in self.borneTemp:
            ind_t1 = borne[0]
            ind_t2 = borne[1]
            res = [idx for idx, val in enumerate(self.psth_time) if val > ind_t1]
            ind_t1=res[0]
            res = [idx for idx, val in enumerate(self.psth_time) if val < ind_t2]
            ind_t2=res[-1]

            for channel in self.psth_compil:
                signal_borne = channel[ind_t1 : ind_t2]
                min_signal = min(signal_borne)
                max_signal = max(signal_borne)
                etendu.append(max_signal - min_signal)
        
            etendu = np.array(etendu)
            etendu = etendu[ind]
            plt.plot(s_courant_val, etendu,"o", label= str(borne[0]) + ":" + str(borne[1]) + " ms")
            plt.plot(s_courant_val, etendu,"-")
            plt.legend(fontsize=10)
            plt.xlabel("Puissance (mW)")
            plt.ylabel("EMG max amplitude (V)")
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            etendu = []
            if (len(saveplotName) >= 1):
                self.saveFigure(saveplotName)
        plt.show()
    
    def peak2peakPulse(self, saveplotName = ""):
        # chan : psth array, borneTemp : borne à extraire peak amp
        dict={}
        etendu = []
        ind = np.argsort(self.pulseWidth)
        s_pulseWidth = np.sort(self.pulseWidth)
        for borne in self.borneTemp:
            ind_t1 = borne[0]
            ind_t2 = borne[1]
            res = [idx for idx, val in enumerate(self.psth_time) if val > ind_t1]
            ind_t1=res[0]
            res = [idx for idx, val in enumerate(self.psth_time) if val < ind_t2]
            ind_t2=res[-1]

            for channel in self.psth_compil:
                signal_borne = channel[ind_t1 : ind_t2]
                min_signal = min(signal_borne)
                max_signal = max(signal_borne)
                etendu.append(max_signal - min_signal)
        
            etendu = np.array(etendu)
            etendu = etendu[ind]
            plt.plot(s_pulseWidth, etendu,"o", label= str(borne[0]) + ":" + str(borne[1]) + " ms")
            plt.plot(s_pulseWidth, etendu,"-")
            plt.legend(fontsize=10)
            plt.xlabel("Pulse Width (ms)")
            plt.ylabel("EMG max amplitude (V)")
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            etendu = []
            if (len(saveplotName) >= 1):
                self.saveFigure(saveplotName)
        plt.show()

    def peak2peakFrequence(self, saveplotName = ""):
        # chan : psth array, borneTemp : borne à extraire peak amp
        dict={}
        etendu = []
        ind = np.argsort(self.frequence)
        s_frequence = np.sort(self.frequence)
        for borne in self.borneTemp:
            ind_t1 = borne[0]
            ind_t2 = borne[1]
            res = [idx for idx, val in enumerate(self.psth_time) if val > ind_t1]
            ind_t1=res[0]
            res = [idx for idx, val in enumerate(self.psth_time) if val < ind_t2]
            ind_t2=res[-1]

            for channel in self.psth_compil:
                signal_borne = channel[ind_t1 : ind_t2]
                min_signal = min(signal_borne)
                max_signal = max(signal_borne)
                etendu.append(max_signal - min_signal)
        
            etendu = np.array(etendu)
            etendu = etendu[ind]
            plt.plot(s_frequence, etendu,"o", label= str(borne[0]) + ":" + str(borne[1]) + " hz")
            plt.plot(s_frequence, etendu,"-")
            plt.legend(fontsize=10)
            plt.xlabel("Frequence train(hz)")
            plt.ylabel("EMG max amplitude (V)")
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            etendu = []
            if (len(saveplotName) >= 1):
                self.saveFigure(saveplotName)
        plt.show()

    def findMxNsubplotGrid(self, x:int):
        "Best subplot display. Return tuple 2 éléments"
        x_square_root_floor = math.floor(math.sqrt(x))
        reste = math.sqrt(x) - x_square_root_floor
        
        if reste >= 0.5:
            mXn = (x_square_root_floor + 1, x_square_root_floor + 1)
        elif reste == 0:
            mXn = (x_square_root_floor, x_square_root_floor)
        else:
            mXn = (x_square_root_floor + 1, x_square_root_floor)
        return mXn
      
    def saveFigure(self, filename):
        pathExt = self.saveFigPathFolder + '/' + filename + ".svg"
        plt.savefig(pathExt)

    def find_latency(self, data, window_size=10, k=0.08):
        """
        Detect breakpoints in a time series using a rolling diff approach.

        Parameters:
        - data: numpy array representing the time series
        - window_size: size of the rolling window for calculating the diff
        - k: factor for the adaptive threshold

        Returns:
        - above_threshold_indices: numpy array of indices where the rolling diff exceeds the threshold
        """

        # Calculate the difference for each point
        diffs = np.diff(data)

        # Apply a rolling mean to the differences
        window = np.ones(window_size) / window_size
        rolling_diffs = np.convolve(diffs, window, mode='valid')

        # Determine the adaptive threshold
        mean_rolling_diff = np.mean(rolling_diffs)
        std_rolling_diff = np.std(rolling_diffs)
        threshold = mean_rolling_diff + k * std_rolling_diff

        # Identify points where the rolling diff exceeds the threshold
        above_threshold_indices = np.where(np.abs(rolling_diffs) > threshold)[0] + window_size - 1
        above_threshold_indices = above_threshold_indices[above_threshold_indices>30]

        return above_threshold_indices[0]
    
    def calibrationForceVoltage(self,ChannelNumber,massValue):
        """Calibration base on weight add to the sensor. Must give the channel number -1 (from labchart) and 
         the corresponding weight by block -1"""
        voltage_moy=[]
        for i in range(len(massValue)):
            value = self.c_data[(ChannelNumber, i)] # data of one block
            valueAve = medfilt(value,101)# mov average on 101 pts
            voltage_moy.append(np.median(valueAve)) 
        result = linregress(voltage_moy, massValue)
        self.calibrationSenseur = {"slope": result.slope, "intercept": result.intercept, "rvalue" : result.rvalue}
    
    def find_latency(self, data, window_size=10, k=0.08):
        """Detect breakpoints in a time series using a rolling diff approach.
    
        Parameters:
        - data: numpy array representing the time series
        - window_size: size of the rolling window for calculating the diff
        - k: factor for the adaptive threshold
    
        Returns:
        - above_threshold_indices: numpy array of indices where the rolling diff exceeds the threshold
        """
    
        # Calculate the difference for each point
        diffs = np.diff(data)
    
        # Apply a rolling mean to the differences
        window = np.ones(window_size) / window_size
        rolling_diffs = np.convolve(diffs, window, mode='valid')
    
        # Determine the adaptive threshold
        mean_rolling_diff = np.mean(rolling_diffs)
        std_rolling_diff = np.std(rolling_diffs)
        threshold = mean_rolling_diff + k * std_rolling_diff
    
        # Identify points where the rolling diff exceeds the threshold
        above_threshold_indices = np.where(np.abs(rolling_diffs) > threshold)[0] + window_size - 1
        above_threshold_indices = above_threshold_indices[above_threshold_indices>30]
    
        return above_threshold_indices[0]
        
    def rectEmg(self, signal, stim_index, train_duration, showPlot, emg_max_latency = 0.15,thresholdMinEmg = 0.01):
        "Rectified EMG compute around specific window. Peak to peak amplitude. Function for train"
        beforeFirst = math.floor(.5 * self.freq)
        firstIndex = stim_index[0] - beforeFirst
        endOfTrain = stim_index[0] + (train_duration * self.freq)
        t = np.arange(endOfTrain - firstIndex)/self.freq
        
        # rectified segment
        sos = butter(2, 50, 'hp', fs=self.freq, output='sos')
        signalFilt = sosfilt(sos, signal) # high pass filter 50hz
        rectSignalFilt = abs(signalFilt)
        sos = butter(2, 10, 'lp', fs=self.freq, output='sos')
        signalFiltFinal = sosfilt(sos, rectSignalFilt) # high pass filter 50hz
        # Area under curve
        dx_time = 1/self.freq
        diff_ind = endOfTrain - stim_index[0]
        areaUnderCurve = sum(signalFiltFinal[stim_index[0] : endOfTrain] * dx_time)*diff_ind 
        # peak to peak and latency
        window_end_bin = math.floor(self.freq * emg_max_latency)
        peakTopeak = []
        latency = []
        areaUnderCurveCompil = [] # par pulse 
        
        for i in stim_index:
            seg = signal[i:(i + window_end_bin)]
            diff_ind = window_end_bin
            temp = sum(seg * dx_time)*diff_ind 
            if (max(seg) > thresholdMinEmg): # Calcul les latences uniquement si il y a emg
                index = math.floor(self.find_latency(seg, window_size=10, k=0.1))
                latency.append(index / self.freq)
                areaUnderCurveCompil.append(temp)
            peakTopeak.append(max(seg)-min(seg))
            
        peakTopeakAverage = sum(peakTopeak) / len(peakTopeak)
        peakTopeakMax = max(peakTopeak)
        if (len(latency)>1):
            latency_ave = sum(latency) / len(latency)
            latency_std = st.stdev(latency)
            latency_min = min(latency)
            latency_max = max(latency)
            
        else : 
            latency_ave = 0
            latency_std = 0
            latency_min = 0
            latency_max = 0
            

        # plotting rectification
        if showPlot:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True)
            ax1.plot(t,signal[firstIndex : endOfTrain])
            ax2.plot(t,rectSignalFilt[firstIndex : endOfTrain])
            ax3.plot(t,signalFiltFinal[firstIndex : endOfTrain])
        
            plt.tight_layout()
            plt.show()
        stats = {"AUC":areaUnderCurve, "AUCParPulse": areaUnderCurveCompil ,"PeakToPeak":peakTopeak,"PeakToPeakAverage":peakTopeakAverage,"PeakToPeakMax":peakTopeakMax, "latencyVal":latency,"latencyAverage":
                 latency_ave,"latency_min":latency_min,"latency_max":latency_max,"latency_std":latency_std,"temps":t,"borneSegment":(firstIndex, endOfTrain),"RectSignal":signalFiltFinal}
        self.statsEmg = stats
        return stats

    def forceAreaUnderCurve(self, force, stim_index, train_duration, showPlot = True):
        "Calcul l'aire sous la courbe de la force générée durant un train, la force filtrée, la force calibrée"
        beforeFirst = math.floor(.5 * self.freq)
        firstIndex = stim_index[0] - beforeFirst
        endOfTrain = stim_index[0] + (train_duration * self.freq)
        t = np.arange(endOfTrain - firstIndex)/self.freq
        
        valueMedFilt = medfilt(force,101) # Pas encore transformé en force
        m = self.calibrationSenseur["slope"] # pente de la calibration force
        b = self.calibrationSenseur["intercept"] # ordonnée a l'origine de la calibration force
        forceFilt = m * valueMedFilt + b
        minForceFilt = min(forceFilt[stim_index[0] : endOfTrain])
        forceFilt -= minForceFilt
        # Area under curve
        dx_time = 1/self.freq
        areaUnderCurve = sum(forceFilt[stim_index[0] : endOfTrain] * dx_time) # have to find what is the real voltage for the units
        
        # plotting voltage and force
        if showPlot:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True)
            ax1.plot(t,force[firstIndex : endOfTrain])
            ax1.plot(t,valueMedFilt[firstIndex : endOfTrain],"r")
            ax2.plot(t,valueMedFilt[firstIndex : endOfTrain])
            ax3.plot(t,forceFilt[firstIndex : endOfTrain],"b")      
            plt.tight_layout()
            plt.show()
        
        stats = {"AUC":areaUnderCurve,"forceRaw":force,"forceMedFilt":valueMedFilt,"forceCalib":forceFilt,"temps":t,"borne":(firstIndex, endOfTrain)}
        self.statsForce = stats
        return stats