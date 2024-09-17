import psth_class as P
import adi
import numpy as np
import matplotlib.pyplot as plt
import data_analysis as da # Vincent's function
from scipy.signal import find_peaks
from scipy.signal import medfilt 
from scipy.stats import linregress
from scipy.signal import butter
from scipy.signal import sosfilt
from scipy import integrate
import os
import math
import tkinter as tk
from tkinter import filedialog

def showBarPlot(x_value, groupValue, figTitre, xLabel, yLabel, yLimite):
    """Produit figures de bar plot
        x_value : tuple des valeurs en absisse. groupValue : dictionnaire les clefs représente le nom de chacun des données regroupée
        par exemple tous les pulse de 1 ms, les valeurs correspondantes sont les valeurs en y. la clef est répétée en x"""

    
    x = np.arange(len(x_value))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for key, count in groupValue.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, count, width, label=key,)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(yLabel)
    ax.set_title(figTitre)
    ax.set_xticks(x + width, x_value)
    ax.set_xlabel(xLabel)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(yLimite)

    plt.show()

def showRawData(psthObject, calibration, limiteSup):
    """Produit figure avec sous graphiques des données brutes à partir de l'objet psth
    l'objet contient tous les blocs, tous les channels
        """
    
    total_bloc = psthObject.adi_out[0].n_records # total des blocs du fichier
    psthObject.calibrationSenseur = calibration.calibrationSenseur # fichier calibration senseur force
    
    for bloc in range(total_bloc):
        signal = psthObject.c_data[(0,bloc)] # premier canal du fichier chart = emg
        stim = psthObject.c_data[(1,bloc)] # 2e canal du fichier = stimulation pulse
        force = psthObject.c_data[(2,bloc)] # 3e canal du fichier = la force non calibrée
        index = da.find_event_index(stim,10000, psthObject.select, psthObject.time_window, psthObject.experiment)
        psthObject.forceAreaUnderCurve(force, index, 2, False) # aire sous la courbe force
        psthObject.rectEmg(signal, index, 2, False) # rectification de l'emg et d'autre analyse
        forceCalib = psthObject.statsForce["forceCalib"]
        t_force = psthObject.statsForce["temps"]
        forceAUC = psthObject.statsForce["AUC"]
        signalRect = psthObject.statsEmg["RectSignal"]
        t_emg = psthObject.statsEmg["temps"]
        debut = psthObject.statsEmg["borneSegment"][0]
        fin =  psthObject.statsEmg["borneSegment"][1]
        signalSeg = signal[debut:fin]
        signalRect = signalRect[debut:fin]
        
        stimSeg = stim[debut:fin]
        debut = psthObject.statsForce["borne"][0]
        fin =  psthObject.statsForce["borne"][1]
        force = force[debut:fin]
        forceCal = psthObject.statsForce["forceCalib"]
        forceCal = forceCal[debut:fin]
        # subplot du ième bloc
        # voir figure subfigure pour faire plusieurs figure dans une
        fig, axs = plt.subplots(5)
        axs[0].plot(t_emg, signalSeg)
        axs[0].set_title('EMG signal')
        axs[0].set_ylim(-limiteSup[0], limiteSup[0])
        axs[1].plot(t_emg, stimSeg)
        axs[1].set_title('Stimulation')
        axs[1].set_ylim(0, limiteSup[1])
        axs[2].plot(t_emg, signalRect)
        axs[2].set_title("EMG rectifié")
        axs[2].set_ylim(-0.5, limiteSup[2])
        axs[3].plot(t_force, force)
        axs[3].set_title("Force (V)")
        axs[3].set_ylim(4.235, limiteSup[3])
        axs[4].plot(t_force, forceCal)
        axs[4].set_title("Force calibrée (g)")
        axs[4].set_ylim(-0.05, limiteSup[4])
        
        plt.show()


if __name__ == "__main__":
  
    # 1 fichier à la fois :

    # Importe les fichiers labchart et les formatent
    
   
    #--------CALIBRATION DU CAPTEUR FORCE----------------
    dir_path = "C:/Users/Maxime/Desktop/FredericD/Calibration/"
    calibration = P.Psth(dir_path, "elect") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
    calibration.loadLabchartFromDir()
    calibration.calibrationForceVoltage(2,[0, 50, 55, 60, 70, 90, 110, 130]) # basé sur calibration faite le 25 juin 24
    m = calibration.calibrationSenseur["slope"] # pente de la calibration force
    b = calibration.calibrationSenseur["intercept"] # ordonnée a l'origine de la calibration force
    
    
    #--------RAT 287----- PARAMÈTRES OPTIMAUX OPTOGÉNÉTIQUES --------------
        # LOAD LE FICHIER pour une stimulation de 300mA
    dir_path = "C:/Users/Maxime/Desktop/FredericD/287/opto_gauche/parametres_optimaux_300/"
    psth2 = P.Psth(dir_path, "elect") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
    psth2.loadLabchartFromDir()
    psth2.calibrationSenseur = calibration.calibrationSenseur
    # showRawData(psth2, calibration, [2.2, 2.2, 2, 4.35, 20])
    #     # ANALYSE FORCE ET EMG CONDITION PAR CONDITION
    total_bloc = psth2.adi_out[0].n_records
    # liste : [numéro bloc, durée pulse, fréquence et puissance laser]
    bloc_informations = \
        [[0, 1, 5, 300],[1, 1, 10, 300], [2, 1, 20, 300],[3, 1, 30, 300], [4, 1, 50, 300], \
        [5, 5, 5, 300], [6, 5, 10, 300], [7, 5, 20, 300],[8, 5, 30, 300], [9, 5, 50, 300], \
        [10, 10, 5, 300], [11, 10, 10, 300], [12, 10, 20, 300], [13, 10, 30, 300], [14, 10, 50, 300], \
        [15, 15, 5, 300], [16, 15, 10, 300], [17, 15, 20, 300], [18, 15, 30, 300], [19, 15, 50, 300], \
        [20, 20, 5, 300], [21, 20, 10, 300], [22, 20, 20, 300], [23, 20, 30, 300], \
        [24, 25, 5, 300], [25, 25, 10, 300], [26, 25, 20, 300], [27, 25, 30, 300]] 
    
    
    forceAUC = []
    emgAUC = []
    emgP2PMax = []
    emgP2PAve = []
    latencyAve = []
    latencyStd = []
    latencyMin = []
    latencyMax = []
    for bloc in range(total_bloc):
        signal = psth2.c_data[(0,bloc)]
        stim = psth2.c_data[(1,bloc)]
        force = psth2.c_data[(2,bloc)]
        index = da.find_event_index(stim,10000, psth2.select, psth2.time_window, psth2.experiment)
        psth2.forceAreaUnderCurve(force, index, 2, False) # aire sous la courbe force
        psth2.rectEmg(signal, index, 2, False) # rectification de l'emg et d'autre analyse
        forceAUC.append(psth2.statsForce["AUC"])
        emgAUC.append(psth2.statsEmg["AUC"])
        emgP2PMax.append(psth2.statsEmg["PeakToPeakMax"])
        emgP2PAve.append(psth2.statsEmg["PeakToPeakAverage"])
        latencyAve.append(psth2.statsEmg["latencyAverage"])
        latencyStd.append(psth2.statsEmg["latency_std"])
        latencyMin.append(psth2.statsEmg["latency_min"])
        latencyMax.append(psth2.statsEmg["latency_max"])
        

    info = np.array(bloc_informations)
    freq = info[:, 2]
    
    # Sauvegarde des variables :
    emgAUC_300mA = emgAUC
    emgP2PMax_300mA = emgP2PMax
    emgP2PAve_300mA = emgP2PAve
    forceAUC_300mA = forceAUC
    freq_train_300mA = freq
    latAve_300mA = latencyAve
    latStd_300mA = latencyStd
    latMin_300mA = latencyMin
    latMax_300mA = latencyMax
    
    # ajout d'élément pour combler l'absence de données les listes doivent contenir le même nombre d'éléments
    latAve_300mA.insert(24,0)
    latAve_300mA.append(0)
    
    # transformation des latences en index en ms :
    temp =(np.array(latAve_300mA)/10).round(2)
    
    latence = {
    '1 ms': (temp[0:5]),
    '5 ms': (temp[5:10]),
    '10 ms': (temp[10:15]),
    '15 ms': (temp[15:20]),
    '20 ms': (temp)[20:25], 
    '25 ms': (temp)[25:30]
    }
    figTitre = "Latence de l'EMG selon différents paramètres pour un laser de 300mA" 
    xLab = "Fréquence du train"
    yLabel = "Latence (ms)"

    # showBarPlot((5, 10, 20, 30, 50), latence, figTitre, xLab,
    #     yLabel, yLimite=(0, 10) 
    #     )

    # plt.plot(freq[0:5], emgAUC[0:5], "b-o", label="1ms")
    # plt.plot(freq[5:10], emgAUC[5:10], "y-o", label="5ms")
    # plt.plot(freq[10:15], emgAUC[10:15], "r-o", label="10ms")
    # plt.plot(freq[15:20], emgAUC[15:20], "k-o", label="15ms")
    # plt.plot(freq[20:24], emgAUC[20:24], "m-o", label="20ms")
    # plt.plot(freq[24:28], emgAUC[24:28], "g-o", label="25ms")
    # plt.xlabel("freq (Hz)")
    # plt.ylim((0, 0.1))
    # plt.ylabel("V*s")
    # plt.title("Aire sous la courbe EMG à 300mA")
    # plt.grid(True, color='gray', linestyle='-.')
    # plt.legend(title="Pulse")
    # plt.show()

     # LOAD LE FICHIER pour une stimulation de 600mA
    dir_path = "C:/Users/Maxime/Desktop/FredericD/287/opto_gauche/parametres_optimaux_600/"
    psth3 = P.Psth(dir_path, "elect") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
    psth3.loadLabchartFromDir()
    psth3.calibrationSenseur = calibration.calibrationSenseur
    # showRawData(psth3, calibration, [2.2, 2.2, 2, 4.35, 20])
        # ANALYSE FORCE ET EMG CONDITION PAR CONDITION
    total_bloc = psth3.adi_out[0].n_records
    # liste : [numéro bloc, durée pulse, fréquence et puissance laser]
    bloc_informations = \
        [[0, 1, 5, 600],[1, 1, 20, 600], [2, 1, 50, 600], \
        [3, 5, 5, 600], [4, 5, 20, 600], [5, 5, 50, 600], \
        [6, 10, 5, 600], [7, 10, 20, 600], [8, 10, 50, 600], \
        [9, 20, 5, 600], [10, 20, 20, 600]] 
    
    
    forceAUC = []
    emgAUC = []
    emgP2PMax = []
    emgP2PAve = []
    latencyAve = []
    latencyStd = []
    latencyMin = []
    latencyMax = []
    
    for bloc in range(total_bloc):
        signal = psth3.c_data[(0,bloc)]
        stim = psth3.c_data[(1,bloc)]
        force = psth3.c_data[(2,bloc)]
        index = da.find_event_index(stim,10000, psth3.select, psth3.time_window, psth3.experiment)
        psth3.forceAreaUnderCurve(force, index, 2, False) # aire sous la courbe force
        psth3.rectEmg(signal, index, 2, False) # rectification de l'emg et d'autre analyse
        forceAUC.append(psth3.statsForce["AUC"])
        emgAUC.append(psth3.statsEmg["AUC"])
        emgP2PMax.append(psth3.statsEmg["PeakToPeakMax"])
        emgP2PAve.append(psth3.statsEmg["PeakToPeakAverage"])
        latencyAve.append(psth3.statsEmg["latencyAverage"])
        latencyStd.append(psth3.statsEmg["latency_std"])
        latencyMin.append(psth3.statsEmg["latency_min"])
        latencyMax.append(psth3.statsEmg["latency_max"])

    info = np.array(bloc_informations)
    freq = info[:, 2]
    
     # Sauvegarde des variables :
    emgAUC_600mA = emgAUC
    emgP2PMax_600mA = emgP2PMax
    emgP2PAve_600mA = emgP2PAve
    forceAUC_600mA = forceAUC
    freq_train_600mA = freq
    latAve_600mA = latencyAve
    latStd_600mA = latencyStd
    latMin_600mA = latencyMin
    latMax_600mA = latencyMax
    # ajout d'élément pour combler l'absence de données les listes doivent contenir le même nombre d'éléments
    latAve_600mA.append(0)
    
    # transformation des latences en index en ms :
    temp =(np.array(latAve_600mA)/10).round(2)
    
    latence = {
    '1 ms': (temp[0:3]),
    '5 ms': (temp[3:6]),
    '10 ms': (temp[6:9]),
    '20 ms': (temp)[9:12] 
    }
    figTitre = "Latence de l'EMG selon différents paramètres pour un laser de 600mA" 
    xLab = "Fréquence du train"
    yLabel = "Latence (ms)"

    showBarPlot((5, 20, 50), latence, figTitre, xLab,
        yLabel, yLimite=(0, 10) 
        )
    # plt.plot(freq[0:3], emgAUC[0:3], "b-o", label="1ms")
    # plt.plot(freq[3:6], emgAUC[3:6], "y-o", label="5ms")
    # plt.plot(freq[6:9], emgAUC[6:9], "r-o", label="10ms")
    # plt.plot(freq[9:11], emgAUC[9:11], "m-o", label="20ms")
    
    # plt.xlabel("freq (Hz)")
    # plt.ylabel("V*s")
    # plt.ylim((0, 0.1))
    # plt.title("Aire sous la courbe EMG à 600mA")
    # plt.grid(True, color='gray', linestyle='-.')
    # plt.legend(title="Pulse")
    # plt.show()
    
    # LOAD LE FICHIER pour une stimulation de 1200mA
    dir_path = "C:/Users/Maxime/Desktop/FredericD/287/opto_gauche/parametres_optimaux_1200/"
    psth4 = P.Psth(dir_path, "elect") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
    psth4.loadLabchartFromDir()
    psth4.calibrationSenseur = calibration.calibrationSenseur
    showRawData(psth4, calibration, [2.2, 2.2, 2, 4.35, 20])

        # ANALYSE FORCE ET EMG CONDITION PAR CONDITION
    total_bloc = psth4.adi_out[0].n_records
    # liste : [numéro bloc, durée pulse, fréquence et puissance laser]
    bloc_informations = \
        [[0, 1, 5, 1200],[1, 1, 20, 1200], [2, 1, 50, 1200], \
        [3, 5, 5, 1200], [4, 5, 20, 1200], [5, 5, 50, 1200], \
        [6, 10, 5, 1200], [7, 10, 20, 1200], [8, 10, 50, 1200], \
        [9, 20, 5, 1200], [10, 20, 20, 1200]] 
    
    
    forceAUC = []
    emgAUC = []
    emgP2PMax = []
    emgP2PAve = []
    latencyMoyenne = []
    latencyAve = []
    latencyStd = []
    latencyMin = []
    latencyMax = []
    
    for bloc in range(total_bloc):
        signal = psth4.c_data[(0,bloc)]
        stim = psth4.c_data[(1,bloc)]
        force = psth4.c_data[(2,bloc)]
        index = da.find_event_index(stim,10000, psth4.select, psth4.time_window, psth4.experiment)
        psth4.forceAreaUnderCurve(force, index, 2, False) # aire sous la courbe force
        psth4.rectEmg(signal, index, force, 2, False) # rectification de l'emg et d'autre analyse
        forceAUC.append(psth4.statsForce["AUC"])
        emgAUC.append(round(psth4.statsEmg["AUC"],3))
        emgP2PMax.append(psth4.statsEmg["PeakToPeakMax"])
        emgP2PAve.append(psth4.statsEmg["PeakToPeakAverage"])
        latencyMoyenne.append(psth4.statsEmg["latencyAverage"])
        latencyAve.append(psth4.statsEmg["latencyAverage"])
        latencyStd.append(psth4.statsEmg["latency_std"])
        latencyMin.append(psth4.statsEmg["latency_min"])
        latencyMax.append(psth4.statsEmg["latency_max"])

    info = np.array(bloc_informations)
    freq = info[:, 2]

     # Sauvegarde des variables :
    emgAUC_1200mA = emgAUC
    emgP2PMax_1200mA = emgP2PMax
    emgP2PAve_1200mA = emgP2PAve
    forceAUC_1200mA = forceAUC
    freq_train_1200mA = freq
    latAve_1200mA = latencyAve
    latStd_1200mA = latencyStd
    latMin_1200mA = latencyMin
    latMax_1200mA = latencyMax
    
    # ajout d'élément pour combler l'absence de données les listes doivent contenir le même nombre d'éléments
    latAve_1200mA.append(0)
    
    # transformation des latences en index en ms :
    temp =(np.array(latAve_1200mA)/10).round(2)
    
    latence = {
    '1 ms': (temp[0:3]),
    '5 ms': (temp[3:6]),
    '10 ms': (temp[6:9]),
    '20 ms': (temp)[9:12] 
    }
    figTitre = "Latence de l'EMG selon différents paramètres pour un laser de 1200mA" 
    xLab = "Fréquence du train"
    yLabel = "Latence (ms)"

    showBarPlot((5, 20, 50), latence, figTitre, xLab,
        yLabel, yLimite=(0, 10) 
        )

    # plt.plot(freq[0:3], emgAUC[0:3], "b-o", label="1ms")
    # plt.plot(freq[3:6], emgAUC[3:6], "y-o", label="5ms")
    # plt.plot(freq[6:9], emgAUC[6:9], "r-o", label="10ms")
    # plt.plot(freq[9:11], emgAUC[9:11], "m-o", label="20ms")
    
    # plt.xlabel("freq (Hz)")
    # plt.ylabel("V*s")
    # plt.ylim((0, 0.1))
    # plt.title("Aire sous la courbe EMG à 1200mA")
    # plt.grid(True, color='gray', linestyle='-.')
    # plt.legend(title="Pulse")
    # plt.show()

    # # Recrutement EMG aire sous la courbe selon puissance laser, fréquence des trains et durée des pulses
    #     # pulse 1 ms
    # plt.plot(freq_train_300mA[0:5], emgAUC_300mA[0:5], "b-o", label="300mA")
    # plt.plot(freq_train_600mA[0:3], emgAUC_600mA[0:3], "k-o", label="600mA")
    # plt.plot(freq_train_1200mA[0:3], emgAUC_1200mA[0:3], "r-o", label="1200mA")
    # plt.xlabel("freq train (Hz)")
    # plt.ylabel("V*s")
    # plt.ylim((0, 0.1))
    # plt.title("Aire sous la courbe EMG selon puissance Laser pour un pulse 1ms")
    # plt.grid(True, color='gray', linestyle='-.')
    # plt.legend(title="Puissance")
    # plt.show()

    #     # pulse 5 ms
    # plt.plot(freq_train_300mA[5:10], emgAUC_300mA[5:10], "b-o", label="300mA")
    # plt.plot(freq_train_600mA[3:6], emgAUC_600mA[3:6], "k-o", label="600mA")
    # plt.plot(freq_train_1200mA[3:6], emgAUC_1200mA[3:6], "r-o", label="1200mA")
    # plt.xlabel("freq train (Hz)")
    # plt.ylabel("V*s")
    # plt.ylim((0, 0.1))
    # plt.title("Aire sous la courbe EMG selon puissance Laser pour un pulse 5ms")
    # plt.grid(True, color='gray', linestyle='-.')
    # plt.legend(title="Puissance")
    # plt.show()

    # # pulse 10 ms
    # plt.plot(freq_train_300mA[10:15], emgAUC_300mA[10:15], "b-o", label="300mA")
    # plt.plot(freq_train_600mA[6:9], emgAUC_600mA[6:9], "k-o", label="600mA")
    # plt.plot(freq_train_1200mA[6:9], emgAUC_1200mA[6:9], "r-o", label="1200mA")
    # plt.xlabel("freq train (Hz)")
    # plt.ylabel("V*s")
    # plt.ylim((0, 0.1))
    # plt.title("Aire sous la courbe EMG selon puissance Laser pour un pulse 10ms")
    # plt.grid(True, color='gray', linestyle='-.')
    # plt.legend(title="Puissance")
    # plt.show()

    # # pulse 20 ms
    # plt.plot(freq_train_300mA[15:20], emgAUC_300mA[15:20], "b-o", label="300mA")
    # plt.plot(freq_train_600mA[9:11], emgAUC_600mA[9:11], "k-o", label="600mA")
    # plt.plot(freq_train_1200mA[9:11], emgAUC_1200mA[9:11], "r-o", label="1200mA")
    # plt.xlabel("freq train (Hz)")
    # plt.ylabel("V*s")
    # plt.ylim((0, 0.1))
    # plt.title("Aire sous la courbe EMG selon puissance Laser pour un pulse 20ms")
    # plt.grid(True, color='gray', linestyle='-.')
    # plt.legend(title="Puissance")
    # plt.show()


# trainFrequency = (5, 20, 50)


# emgAUC_1200mA.append(0.0)

# Emg_rectify = {
#     '1 ms': (emgAUC_1200mA[0:3]),
#     '5 ms': (emgAUC_1200mA[3:6]),
#     '10 ms': (emgAUC_1200mA[6:9]),
#     '20 ms': (emgAUC_1200mA[9:12]),
# }


# x = np.arange(len(trainFrequency))  # the label locations
# width = 0.15  # the width of the bars
# multiplier = 0

# fig, ax = plt.subplots(layout='constrained')

# for pulseTime, count in Emg_rectify.items():
#     offset = width * multiplier
#     rects = ax.bar(x + offset, count, width, label=pulseTime,)
#     ax.bar_label(rects, padding=3)
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Rect Emg (|V|)')
# ax.set_title('Réponse EMG en fonctions de différents paramètres')
# ax.set_xticks(x + width, trainFrequency)
# ax.set_xlabel("Train frequency (Hz)")
# ax.legend(loc='upper left', ncols=3)
# ax.set_ylim(0, 0.1)

# plt.show()