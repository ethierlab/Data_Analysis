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
import pandas as pd

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
        self.borneTemp = [[0.0015, 0.025]] # self.borneTemp = [[0, 0.0075], [0.0075, 0.015], [0, 0.025]] pour 3 bornes sur le meme graphique
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
            Extract the stimulation curent amplitude value in the file name, base on pre and post File character "_". More than one file
            each file only one block 
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

        if (postFileNameChara == "ma") or (postFileNameChara == "ua"):
            if n_files == 0:
                print("Error no files to analyse")
            elif n_files == 1:
                if files[0].endswith(".adicht"):
                    txt_path = os.path.join(self.path, files[0])
                    ind1 = files[0].rfind(preFileNameChara)
                    ind2 = files[0].rindex(postFileNameChara)
                    courant_val.append(float(files[0][ind1+1:ind2]))
                    f = adi.read_file(txt_path)
                    adi_out.append(f)
                else:
                    print("Error no labchart files to analyse")

            else:
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

            if n_files == 0:
                print("Error no files to analyse")
            elif n_files == 1:
                if files[0].endswith(".adicht"):
                    txt_path = os.path.join(self.path, files[0])
                    ind1 = files[0].rfind(preFileNameChara)
                    ind2 = files[0].rindex(postFileNameChara)
                    pulse.append(float(files[0][ind1+1:ind2]))
                    f = adi.read_file(txt_path)
                    adi_out.append(f)
                else:
                    print("Error no labchart files to analyse")

            else:
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
            # condition variable indépendante : intensité hz
            frequence = []
            adi_out = []

            if n_files == 0:
                print("Error no files to analyse")
            elif n_files == 1:
                if files[0].endswith(".adicht"):
                    txt_path = os.path.join(self.path, files[0])
                    ind1 = files[0].rfind(preFileNameChara)
                    ind2 = files[0].rindex(postFileNameChara)
                    frequence.append(float(files[0][ind1+1:ind2]))
                    f = adi.read_file(txt_path)
                    adi_out.append(f)
                else:
                    print("Error no labchart files to analyse")

            else:
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
        Plusieurs bloc dans un seul fichier labchart.
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
        
        if n_files == 0:
            print("Error no files to analyse")
        elif n_files == 1: # one file available
            if files[0].endswith(".adicht"):
                txt_path = os.path.join(self.path, files[0])
                f = adi.read_file(txt_path)
                adi_out.append(f) # Output of adi read
            else:
                print("Error no labchart files to analyse")
        for i in range(len(adi_out)):
            record = adi_out[i]
            for channel in range(record.n_channels):
                for bloc in range(record.n_records): # au minimum un bloc
                    self.c_data[(channel,bloc)] =  record.channels[channel].get_data(bloc+1)
                    
            
        self.adi_out = adi_out
        
    def fromChannel2Psth(self, t_inf, t_supp, numberSignal, numberEvent, OnePulsePerEvent = True):
        """t_inf, t_supp : time in second before and after the event
            numberSignal : channel number, numberEvent : event channel number
            OnePulsePerEvent : True = each event is not a train. False = each event is a train, first spike took as onset
           
        """

        min_max=[0,0]
        psth_compil = []
        psth_compil_min = []
        psth_compil_max = []
        dataTramePsthSample =  pd.DataFrame()
        etendu=[]
        for i in range(len(self.adi_out)):
            record = self.adi_out[i]
            replication = 0
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
            # Nouveau tableau dataframe
            # construction du dataframe panda :
            for s in sample:
                n = len(s)
                t = np.arange(n)
                t  = (t/self.freq) - t_inf
                min_signal = min(s)
                max_signal = max(s)
                etendu.append([self.courant_val[i], replication+1, max_signal - min_signal])
                listeCourant = np.ones(n)*self.courant_val[i]
                repetition= np.ones(n)*replication+1
                replication += 1
                listParametre = list(zip(listeCourant, t, s, repetition))
                dataTramePsthSample = pd.concat([dataTramePsthSample, pd.DataFrame(listParametre, columns = ["Courant", "Temps", "Segment Psth", "Repetition"])])
                
            
             
            min_psth, moy_psth, max_psth = da.PSTH(sample)
            mini = min(moy_psth)
            maxi = max(moy_psth)
            min_max = [min([min_max[0], mini]), max([min_max[1], maxi])]
            if (len(self.adi_out)>1):
                psth_compil_min.append(min_psth)
                psth_compil.append(moy_psth)
                psth_compil_max.append(max_psth)
            else:
                psth_compil_min = [min_psth]
                psth_compil = [moy_psth]
                psth_compil_max = [max_psth]
        self.dataFrameSegmentPsth = dataTramePsthSample
        self.t_inf = t_inf
        self.t_supp = t_supp
        self.psth_compil = psth_compil
        self.psth_compil_min = psth_compil_min
        self.psth_compil_max = psth_compil_max
        self.min_max = min_max
        self.etenduTouSegment = etendu
    
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
            self.rectEmg(signal_channel, index, 2, showPlot = False, emg_max_latency = 0.025, thresholdMinEmg = 0.1)
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

    def fromChannel2PsthIntraTrainExp3(self, numberSignal, numberEvent, frequenceTrain, dureePsth = 15, intervallePsth = 30, plot = False):
        """ Fonction qui analyse la fatigue d'un pulse répété sur une longue durée à une fréquence constante. 
            numberSignal : channel number, numberEvent : event channel number
            
           
        """
        # Declaration variables
        maxForPlot = 0
        maxForPlotRect = 0
        amplitudeEmgMoyParFreq = {} # dictionnaire des outputs analysés
        self.frequence = frequenceTrain
        min_max = [0,0]
        psth_compil=[]
        
        #--------------- Attribution variable des canaux d'enr et des indices des événements stimulus
        signal_channel=self.c_data[(numberSignal - 1, 0)] # Value associate to the channel bloc
        event_channel=self.c_data[(numberEvent - 1, 0)] # Canal événement associé à un bloc donné, ici un seul bloc dans labchart
        signal_channel = signal_channel / self.signal_channel_gain
        # rectified segment
        sos = butter(2, 50, 'hp', fs=self.freq, output='sos')
        signalFilt = sosfilt(sos, signal_channel) # high pass filter 50hz
        rectSignalFilt = abs(signalFilt)
        sos = butter(2, 10, 'lp', fs=self.freq, output='sos')
        signalFiltFinal = sosfilt(sos, rectSignalFilt) # high pass filter 50hz # SIGNAL RECTIFIÉ
        time = np.arange(len(signal_channel))
        time  = time/self.freq
        indexPulseEvent = da.find_event_index(event_channel, self.freq, self.select, self.time_window, self.experiment)
        

        # ---------------------Amplitude EMG selon le temps ou la nième stimulation----------------------------------
            
        sample = da.cut_individual_event(0, 0.02, indexPulseEvent, signal_channel, self.freq) # sample = psth unique de chaque pulse
        sampleEMGRect =  da.cut_individual_event(0, 0.02, indexPulseEvent, signalFiltFinal, self.freq) # sample = segment rectifié de l'emg
        amplitudeEmg =[]
        amplitudeEmgRect=[]
        for emgUnique in sample:
            amplitudeEmg.append(max(emgUnique)-min(emgUnique))
        for emgRect in sampleEMGRect:
            amplitudeEmgRect.append(max(emgRect))
        
        # -------------------- Figure amplitude EMG selon temps
        # plt.plot(time[indexPulseEvent], medfilt(amplitudeEmg,51))
        # plt.title("Diminution de l'amplitude des EMGs suivant un stimuli répété à " + str(frequenceTrain) + " hz")
        # plt.xlabel("temps (s)")
        # plt.ylabel("Amplitdue EMG (V)")
        # plt.show()



        #----Psth évolutif tout au long des stimulations répétées 
        "Doit déterminer le nombre d'événements choisi par psth, l'intervalle entre les psths ou le nombre de psth "
        dureePsth = dureePsth # duree du psth en seconde
        binPsth = math.floor(dureePsth*self.freq)
        intervallePsth = intervallePsth # intervalle en seconde entre les psths
        binIntervallePsth = math.floor(intervallePsth*self.freq)

        #Structure des trains
        tFin = 0
        indexDebPsth = [indexPulseEvent[0]]
        

        while tFin < time[-1]:
            indPlusGrand = indexDebPsth[-1] + binPsth + binIntervallePsth
            prochainIndex = np.where(indexPulseEvent >= indPlusGrand)[0]
            
            if len(prochainIndex) > 0:
                prochainIndex = indexPulseEvent[prochainIndex[0]]
                indexDebPsth.append(prochainIndex) # Variable debut psth
                tFin = (prochainIndex)/self.freq
            else:
                tFin = time[-1]+12
        
        # Calcul des moyennes et écart-types des amplitudes des EMGs pour la période des psths
        #amplitudeEmg
        #amplitudeEmgRect
        moyPsthAmplitude=[]
        stdPsthAmplitude=[]
        indexDebCompil=[]
        psthTraceX=[] # Psth time
        psthTraceY=[] # y value minimum of the graph
        npoint=20

        for t_index in indexDebPsth:
            binDebut = t_index
            binFin = t_index + (dureePsth * self.freq)
            indexDeb = np.where(indexPulseEvent >= binDebut)
            indexFin = np.where(indexPulseEvent >= binFin)
            
            if indexPulseEvent[-1] >= binFin:
                indexDeb1 = indexDeb[0][0] # au sampling des pulses de stims
                
                indexFin1 = indexFin[0][0] 
                positionDeb = indexPulseEvent[indexDeb1]
                positionFin = indexPulseEvent[indexFin1]
                
                emgPeak = amplitudeEmg[indexDeb1:indexFin1]
                moyenne = np.array(emgPeak).mean()
                std = np.array(emgPeak).std()
                moyPsthAmplitude.append(moyenne)
                stdPsthAmplitude.append(std)
                indexDebCompil.append(positionDeb) # à l'échelle de la freq d'échantillonnage
                psthTraceX.append(np.linspace(time[positionDeb],time[positionFin],npoint))
                psthTraceY.append(np.ones(npoint)*moyenne)
                   
        
        
        if plot:
            plt.figure(1)
            plt.plot(psthTraceX, psthTraceY, '|', color ='gray')
            plt.ylim([0, 2])
            plt.errorbar(time[indexDebCompil], moyPsthAmplitude, stdPsthAmplitude, fmt="|", color ="gray")
            
            plt.figure(2)
            plt.plot(time[indexPulseEvent], medfilt(amplitudeEmg,51),".",color = "gray")
            plt.ylim([0, 2])
            plt.title("Diminution de l'amplitude des EMGs suivant un stimuli répété à " + str(frequenceTrain) + " hz")
            plt.xlabel("temps (s)")
            plt.ylabel("Amplitdue EMG (V)")
            plt.show()
                
    def enveloppe(self, signal):
        "Trouve l'enveloppe des emgs"
        sos = butter(2, 50, 'hp', fs=self.freq, output='sos')
        signalFilt = sosfilt(sos, signal) # high pass filter 50hz
        rectSignalFilt = abs(signalFilt)
        sos = butter(2, 10, 'lp', fs=self.freq, output='sos')
        signEnveloppe = sosfilt(sos, rectSignalFilt) # high pass filter 50hz # SIGNAL RECTIFIÉ
        
        return signEnveloppe

    def fromChannel2PsthIntraTrainExp4(self, t_inf, t_supp, numberSignal, numberEvent, plot):
        """ Fonction qui analyse la fatigue intra train selon la fréquence des pulses intra train. 
        t_inf, t_supp : time in second before and after the event
            numberSignal : channel number, numberEvent : event channel number
            OnePulsePerEvent : True = each event is not a train. False = each event is a train, first spike took as onset
           
        """

        maxForPlot = 0
        maxForPlotRect = 0
        amplitudeEmgMoyParFreq = {} # dictionnaire des outputs analysés
        psth_compil = []
        self.min_max=[0,0]

        for fichier in range(len(self.adi_out)): # boucle fichier par fichier
            
            record = self.adi_out[fichier]
            for channel in range(record.n_channels):
                for bloc in range(record.n_records):
                    self.c_data[(channel,bloc)] =  record.channels[channel].get_data(bloc+1)
                    
            signal_channel=self.c_data[(numberSignal - 1, 0)] # Value associate to the channel bloc
            event_channel=self.c_data[(numberEvent - 1, 0)] # Value associate to the channel bloc
            signal_channel = signal_channel / self.signal_channel_gain
            # rectified segment
            sos = butter(2, 50, 'hp', fs=self.freq, output='sos')
            signalFilt = sosfilt(sos, signal_channel) # high pass filter 50hz
            rectSignalFilt = abs(signalFilt)
            sos = butter(2, 10, 'lp', fs=self.freq, output='sos')
            signalFiltFinal = sosfilt(sos, rectSignalFilt) # high pass filter 50hz # signal rectifié
            time = np.arange(len(signal_channel))
            time  = time/self.freq
            indexPulseEvent = da.find_event_index(event_channel, self.freq, self.select, self.time_window, self.experiment)
            indexTrainEvent = da.takeFirstPeak(indexPulseEvent, .25, .5, self.freq) # second argument = min inter train and 3e = max intra train
            
            # Attribution de la variable psth_compil en vue de produire la figure allpsth
            sample = da.cut_individual_event(t_inf, t_supp, indexTrainEvent, signal_channel, self.freq)
            min_psth, moy_psth, max_psth = da.PSTH(sample)
            psth_compil.append(moy_psth)
            mini = min(moy_psth)
            maxi = max(moy_psth)
            self.min_max = [min([self.min_max[0], mini]), max([self.min_max[1], maxi])]


            # Calcul de l'amplitude du MEP sur une fenêtre restreinte de chaque stimulus. La position du pulse dans le train est considérée
            # Structure des trains :
            structTrain = []
            indexArray = np.array(indexPulseEvent)
            indexTrainEvent.append(indexPulseEvent[-1]+10000)
            
            for i in range(len(indexTrainEvent)-1):
                deb = indexTrainEvent[i]
                fin = indexTrainEvent[i+1]
                linVect = np.where(np.logical_and(indexArray >= deb, indexArray < fin))
                structTrain.append(indexPulseEvent[linVect])
                
                
            # Chaque rangée (val) de structTrain représente un train du fichier courant (à une fréquence donnée)
            amplitudeEmgConcat = []
            amplitudeEmgRectConcat = []
            
            
            for val in structTrain:
                sample = da.cut_individual_event(0, 0.02, val, signal_channel, self.freq) # sample = psth unique de chaque pulse
                sampleEMGRect =  da.cut_individual_event(0, 0.02, val, signalFiltFinal, self.freq) # sample = segment rectifié de l'emg
                amplitudeEmg =[]
                amplitudeEmgRect=[]
                for emgUnique in sample:
                    amplitudeEmg.append(max(emgUnique)-min(emgUnique))
                for emgRect in sampleEMGRect:
                    amplitudeEmgRect.append(max(emgRect))

                amplitudeEmgConcat.append(amplitudeEmg)
                amplitudeEmgRectConcat.append(amplitudeEmgRect)

            # EMG
            nTrainXPulse = np.array(amplitudeEmgConcat).shape
            meanAmpEmgConcat = np.array(amplitudeEmgConcat).mean(0) # moyenne des amplitudes des emgs par train
            stdAmpEmgConcat = np.array(amplitudeEmgConcat).std(0) # standart deviation
            maxForPlot = max([max(meanAmpEmgConcat) , maxForPlot]) # maximum des emgs pour uniformiser graphique 
            amplitudeEmgMoyParFreq.update({(self.frequence[fichier], 'moyenne'): meanAmpEmgConcat})
            amplitudeEmgMoyParFreq.update({(self.frequence[fichier], 'std'): stdAmpEmgConcat })
            amplitudeEmgMoyParFreq.update({(self.frequence[fichier], 'shape'): nTrainXPulse }) # tuple (train,n pulses)

            # EMG Rectifié
            meanAmpEmgRectConcat = np.array(amplitudeEmgRectConcat).mean(0) # moyenne des amplitudes des emgs rect par train
            stdAmpEmgRectConcat = np.array(amplitudeEmgRectConcat).std(0) # standart deviation rect
            maxForPlotRect = max([max(meanAmpEmgRectConcat) , maxForPlotRect]) # maximum des emgs rect pour uniformiser graphique 
            amplitudeEmgMoyParFreq.update({(self.frequence[fichier], 'moyenneRect'): meanAmpEmgRectConcat})
            amplitudeEmgMoyParFreq.update({(self.frequence[fichier], 'stdRect'): stdAmpEmgRectConcat })

            

        if plot :
            subplot_mxn =  self.findMxNsubplotGrid(fichier)
            fin = fichier
            # ordre des graphique en freq ascendante
            ind = np.argsort(self.frequence)
            for i in range(fin):
                expression = "plt.subplot(" + str(subplot_mxn[0]) +","+ str(subplot_mxn[1])+"," + str(i+1) + ")"
                exec(expression)
                x_pulse = amplitudeEmgMoyParFreq[(self.frequence[ind[i]],'shape')][1] # nombre de pulse par train
                x_pulse = range(1, x_pulse + 1)
                x_train = amplitudeEmgMoyParFreq[(self.frequence[ind[i]],'shape')][0] # nombre de train par fichier
                amplitudeEmgMoyen = amplitudeEmgMoyParFreq[(self.frequence[ind[i]], 'moyenne')]
                cerr = amplitudeEmgMoyParFreq[(self.frequence[ind[i]], 'std')]
                
                plt.plot(x_pulse, amplitudeEmgMoyen, ".k" ,label= "Fréquence : " + str(self.frequence[ind[i]]) + " hz" + "n train = " + str(x_train))
                plt.errorbar(x_pulse, amplitudeEmgMoyen, cerr, fmt = ".k")
                plt.xlabel("nième pulse", fontsize=8)
                plt.ylabel("EMG amplitude (V)", fontsize=8)
                plt.legend(fontsize=8)
                plt.ylim((0, maxForPlot + 0.2*maxForPlot))
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
            plt.suptitle("Amplitude et variances des EMG selon sa position au sein du train")
            plt.show()

            # EMG rectifié
            subplot_mxn =  self.findMxNsubplotGrid(fichier)
            fin = fichier
            # ordre des graphique en freq ascendante
            ind = np.argsort(self.frequence)
            for i in range(fin):
                expression = "plt.subplot(" + str(subplot_mxn[0]) +","+ str(subplot_mxn[1])+"," + str(i+1) + ")"
                exec(expression)
                x_pulse = amplitudeEmgMoyParFreq[(self.frequence[ind[i]],'shape')][1] # nombre de pulse par train
                x_pulse = range(1, x_pulse + 1)
                x_train = amplitudeEmgMoyParFreq[(self.frequence[ind[i]],'shape')][0] # nombre de train par fichier
                amplitudeEmgMoyen = amplitudeEmgMoyParFreq[(self.frequence[ind[i]], 'moyenneRect')]
                cerr = amplitudeEmgMoyParFreq[(self.frequence[ind[i]], 'stdRect')]

                plt.plot(x_pulse, amplitudeEmgMoyen, ".k" ,label= "Fréquence : " + str(self.frequence[ind[i]]) + " hz" + "n train = " + str(x_train))
                plt.errorbar(x_pulse, amplitudeEmgMoyen, cerr, fmt = ".k")
                plt.xlabel("nième pulse", fontsize=8)
                plt.ylabel("EMG rectifié (V)", fontsize=8)
                plt.legend(fontsize=8)
                plt.ylim((0, maxForPlotRect + 0.2*maxForPlotRect))
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
            plt.show()
        self.psth_compil = psth_compil
        self.t_inf = t_inf
        self.t_supp = t_supp

    def fromChannel2PsthIntraTrainExp5(self, numberSignal, numberEvent, frequenceTrain, plot):
        """ Fonction qui analyse la fatigue intra train selon une seule fréquence répétée n fois train. 
            numberSignal : channel number, numberEvent : event channel number
            OnePulsePerEvent : True = each event is not a train. False = each event is a train, first spike took as onset
           
        """
        # Declaration variables
        maxForPlot = 0
        maxForPlotRect = 0
        amplitudeEmgMoyParFreq = {} # dictionnaire des outputs analysés
        self.frequence = frequenceTrain
        min_max = [0,0]
        psth_compil=[]
        
        #--------------- Attribution variable des canaux d'enr et des indices des événements stimulus
        signal_channel=self.c_data[(numberSignal - 1, 0)] # Value associate to the channel bloc
        event_channel=self.c_data[(numberEvent - 1, 0)] # Value associate to the channel bloc
        signal_channel = signal_channel / self.signal_channel_gain
        # rectified segment
        sos = butter(2, 50, 'hp', fs=self.freq, output='sos')
        signalFilt = sosfilt(sos, signal_channel) # high pass filter 50hz
        rectSignalFilt = abs(signalFilt)
        sos = butter(2, 10, 'lp', fs=self.freq, output='sos')
        signalFiltFinal = sosfilt(sos, rectSignalFilt) # high pass filter 50hz # SIGNAL RECTIFIÉ
        time = np.arange(len(signal_channel))
        time  = time/self.freq
        indexPulseEvent = da.find_event_index(event_channel, self.freq, self.select, self.time_window, self.experiment)
        indexTrainEvent = da.takeFirstPeak(indexPulseEvent, .25, .5, self.freq) # second argument = min inter train and 3e = max intra train
        #----------------

        #-----------Psth de chacun des trains--------------------------
        sample = da.cut_individual_event(.1, 2, indexTrainEvent, signal_channel, self.freq)
        
        mini = 0
        maxi = 0
        min_max = [min([min_max[0], mini]), max([min_max[1], maxi])]
        
        for val in sample:
            mini = min(val)
            maxi = max(val)
            min_max = [min([min_max[0], mini]), max([min_max[1], maxi])]
            psth_compil.append(val)

        self.psth_compil = psth_compil # psth n'est pas moyenné, il est de format n rangées par bin colonnes
        self.min_max = min_max
        

        # Calcul de l'amplitude du MEP sur une fenêtre restreinte de chaque stimulus. La position du pulse dans le train est considérée
        # Structure des trains :
        structTrain = []
        indexArray = np.array(indexPulseEvent)
        indexTrainEvent.append(indexPulseEvent[-1]+10000)
            
        for i in range(len(indexTrainEvent)-1):
            deb = indexTrainEvent[i]
            fin = indexTrainEvent[i+1]
            linVect = np.where(np.logical_and(indexArray >= deb, indexArray < fin))
            structTrain.append(indexPulseEvent[linVect])
                
                
        # Chaque rangée (val) de structTrain représente un train du fichier courant (à une fréquence donnée)
        amplitudeEmgConcat = []
        amplitudeEmgRectConcat = []
            
            
        for val in structTrain:
            sample = da.cut_individual_event(0, 0.02, val, signal_channel, self.freq) # sample = psth unique de chaque pulse
            sampleEMGRect =  da.cut_individual_event(0, 0.02, val, signalFiltFinal, self.freq) # sample = segment rectifié de l'emg
            amplitudeEmg =[]
            amplitudeEmgRect=[]
            for emgUnique in sample:
                amplitudeEmg.append(max(emgUnique)-min(emgUnique))
            for emgRect in sampleEMGRect:
                amplitudeEmgRect.append(max(emgRect))

            amplitudeEmgConcat.append(amplitudeEmg)
            amplitudeEmgRectConcat.append(amplitudeEmgRect)

        # EMG
        nTrainXPulse = np.array(amplitudeEmgConcat).shape
        meanAmpEmgConcat = np.array(amplitudeEmgConcat).mean(0) # moyenne des amplitudes des emgs par train
        stdAmpEmgConcat = np.array(amplitudeEmgConcat).std(0) # standart deviation
        maxForPlot = max([max(meanAmpEmgConcat) , maxForPlot]) # maximum des emgs pour uniformiser graphique 
        amplitudeEmgMoyParFreq.update({(self.frequence, 'moyenne'): meanAmpEmgConcat})
        amplitudeEmgMoyParFreq.update({(self.frequence, 'std'): stdAmpEmgConcat })
        amplitudeEmgMoyParFreq.update({(self.frequence, 'shape'): nTrainXPulse }) # tuple (train,n pulses)

        # EMG Rectifié
        meanAmpEmgRectConcat = np.array(amplitudeEmgRectConcat).mean(0) # moyenne des amplitudes des emgs rect par train
        stdAmpEmgRectConcat = np.array(amplitudeEmgRectConcat).std(0) # standart deviation rect
        maxForPlotRect = max([max(meanAmpEmgRectConcat) , maxForPlotRect]) # maximum des emgs rect pour uniformiser graphique 
        amplitudeEmgMoyParFreq.update({(self.frequence, 'moyenneRect'): meanAmpEmgRectConcat})
        amplitudeEmgMoyParFreq.update({(self.frequence, 'stdRect'): stdAmpEmgRectConcat })

            

        if plot :
            x_pulse = amplitudeEmgMoyParFreq[(self.frequence,'shape')][1] # nombre de pulse par train
            x_pulse = range(1, x_pulse + 1)
            x_train = amplitudeEmgMoyParFreq[(self.frequence,'shape')][0] # nombre de train par fichier
            amplitudeEmgMoyen = amplitudeEmgMoyParFreq[(self.frequence, 'moyenne')]
            cerr = amplitudeEmgMoyParFreq[(self.frequence, 'std')]

            plt.plot(x_pulse, amplitudeEmgMoyen, ".k" ,label= "Fréquence : " + str(self.frequence) + " hz" + "n train = " + str(x_train))
            plt.errorbar(x_pulse, amplitudeEmgMoyen, cerr, fmt = ".k")
            plt.xlabel("nième pulse", fontsize=8)
            plt.ylabel("EMG amplitude (V)", fontsize=8)
            plt.legend(fontsize=8)
            plt.ylim((min(cerr), maxForPlot + max(cerr) + 0.2*max(cerr)))
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.title("Amplitudes et variances des EMGs selon la position au sein du train")
            plt.show()

            # EMG rectifié
            x_pulse = amplitudeEmgMoyParFreq[(self.frequence,'shape')][1] # nombre de pulse par train
            x_pulse = range(1, x_pulse + 1)
            x_train = amplitudeEmgMoyParFreq[(self.frequence,'shape')][0] # nombre de train par fichier
            amplitudeEmgMoyen = amplitudeEmgMoyParFreq[(self.frequence, 'moyenneRect')]
            cerr = amplitudeEmgMoyParFreq[(self.frequence, 'stdRect')]

            plt.plot(x_pulse, amplitudeEmgMoyen, ".k" ,label= "Fréquence : " + str(self.frequence) + " hz" + "n train = " + str(x_train))
            plt.errorbar(x_pulse, amplitudeEmgMoyen, cerr, fmt = ".k")
            plt.xlabel("nième pulse", fontsize=8)
            plt.ylabel("EMG rectifié (V)", fontsize=8)
            plt.legend(fontsize=8)
            plt.ylim((min(cerr), maxForPlotRect + max(cerr) + 0.2*max(cerr)))
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.title("Amplitudes et variances des EMGs rectifiés selon la position au sein du train")
            plt.show()
 
    def latence(self, OnePulsePerEvent, window, step, tolerance, verif_plot):
        """calcul la latence
        """
        indice=[]
        val_stimulation = [] # repetition de la valeur du courant de stimulation
        niemeRep = []
        emg_amplitude = []   
        for fichier in range(len(self.adi_out)): # Loop fichier par fichier : 1 fichier = 1 intensité dans ce cas ci
            val_courant = self.courant_val[fichier] # établi dans la fonction "loadDataFromDir"
            signal_channel=self.c_data[(0, 0)] # Value associate to the channel bloc
            event_channel=self.c_data[(1, 0)] # Value associate to the channel bloc
            signal_channel = signal_channel / self.signal_channel_gain
            index = da.find_event_index(event_channel, self.freq, self.select, self.time_window, self.experiment)
            if not OnePulsePerEvent:
                index = da.takeFirstPeak(index, .2, .5, self.freq)
            
            emg_max_latency = 0.050
            thresholdMinEmg = 0.01
            dx_time = 1/self.freq
            
            #latency
            window_end_bin = math.floor(self.freq * emg_max_latency)
            
            u=0
            for i in index:
                seg = signal_channel[i - window_end_bin:i + window_end_bin]
                
                if (max(seg) > thresholdMinEmg): # Calcul les latences uniquement si il y a emg
                    # boucle pour trouver l'occurence du premier bin qui est au dessus du threshold
                    depart=0
                    
                    while (depart + window) < (len(seg)):
                        seg2analysis = seg[depart:depart + window]
                        seg2analysis = abs(seg2analysis)
                        med = np.median(seg2analysis)
                        std = np.std(seg2analysis)
                        threshold = med + (tolerance*std)
                        index0 = np.argwhere(seg2analysis>threshold)
                        index1 = (index0 + depart) - window_end_bin
                        index2 = index1/self.freq


                        
                        
                        if (len(index2)>0 and index2[0] >0):
                            niemeRep.append(u)
                            indice.append((index2[0]).tolist())
                            val_stimulation.append(val_courant)
                            if verif_plot:
                                plt.plot(seg)
                                plt.plot(index0[0]+depart,seg[index0[0]+depart],"r+")
                                plt.show()
                            depart = len(seg) # pour sortir du while et n'avoir que la première valeure
                            emg_amplitude.append(max(seg)-min(seg))
                            
                            
                        depart += step
                u+=1
            self.val_courant = np.ravel(val_stimulation)
            self.indice_latence = np.ravel(indice)
            self.niemeRep = niemeRep
            self.emg_amplitude = emg_amplitude
            

            
    
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
                    ind = self.find_latency(seg, window_size=12, k=0.1)
                    
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
    
    def showAllPsthFatigueTrainExp5(self, saveplotName = ""):
        "Plot each Psth of each train and the average Psth"
        u = -1
        subplot_mxn =  self.findMxNsubplotGrid(len(self.psth_compil))
        for psth in self.psth_compil:
            u += 1  
            expression = "plt.subplot(" + str(subplot_mxn[0]) +","+ str(subplot_mxn[1])+"," + str(u+1) + ")"
            exec(expression)
            # puissance = self.calibCourantPuissance(self.courant_val, self.typeStim). Si on veut la calibration finale en Y
            time = np.arange(len(psth))
            time = (time/self.freq)-self.t_inf
            plt.plot(time, psth, label= "Train #" + str(u +1))
            plt.xlabel("time (s)", fontsize=8)
            plt.ylabel("EMG (V)", fontsize=8)
            plt.legend(fontsize=8)
            plt.ylim((self.min_max[0] - self.min_max[0]*.1 ,self.min_max[1]+self.min_max[1]*.1))
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
        plt.suptitle("Psth de chacun des trains à une fréquence de " + str(self.frequence) + " hz") # titre de la figure 
        plt.show()
        
        meanPSTH = np.array(self.psth_compil).mean(0)
        plt.plot(time, meanPSTH, label= "Fréquence :" + str(self.frequence))
        plt.xlabel("time (s)", fontsize=8)
        plt.ylabel("EMG (V)", fontsize=8)
        plt.legend(fontsize=8)
        plt.ylim((self.min_max[0] - self.min_max[0]*.1 ,self.min_max[1]+self.min_max[1]*.1))
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.suptitle("PSTH moyen, stimulation " + self.typeStim) # titre de la figure 
        plt.show()
        
        self.psth_time = time

    def showAllPsthFatigueTrainExp4(self, saveplotName = ""):
        "Plot each Psth of each train and the average Psth"
        
        subplot_mxn =  self.findMxNsubplotGrid(len(self.psth_compil))
        for i in range(len(self.psth_compil)):
            expression = "plt.subplot(" + str(subplot_mxn[0]) +","+ str(subplot_mxn[1])+"," + str(i+1) + ")"
            exec(expression)
            ind = np.argsort(self.frequence)
            # puissance = self.calibCourantPuissance(self.courant_val, self.typeStim). Si on veut la calibration finale en Y
            time = np.arange(len(self.psth_compil[0]))
            time = (time/self.freq)-self.t_inf
            plt.plot(time, self.psth_compil[ind[i]], label= "Fréquence des trains " + str(self.frequence[ind[i]]))
            plt.xlabel("time (s)", fontsize=8)
            plt.ylabel("EMG (V)", fontsize=8)
            plt.legend(fontsize=8)
            plt.ylim((self.min_max[0] - self.min_max[0]*.1 , self.min_max[1]+self.min_max[1]*.1))
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
        plt.suptitle("Psth des stimulations " + self.typeStim + " selon la fréquence des trains ") # titre de la figure 
        plt.show()
                
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
        p0 = [max(ydata), np.median(xdata),1,0] # les paramètres initiaux.[L, x0, k, b] : L le max de la courbe, b : offset en y, k scaling input, x0 :
        # la moitié du output
        try :
            popt, pcov = curve_fit(self.sigmoid, xdata, ydata, p0, method='lm')
            self.paraSigmoid = popt
        except RuntimeError :
            self.paraSigmoid = [0, 0, 0, 0]     
        
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
            self.dataCourbeRecru = [s_courant_val, etendu]
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
    
    def find_latency(self, data, window_size=5, k=0.01):
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
