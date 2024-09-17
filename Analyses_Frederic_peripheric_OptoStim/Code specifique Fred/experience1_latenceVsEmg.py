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

def showRawData(psthObject, calibration):
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
        
        axs[1].plot(t_emg, stimSeg)
        axs[1].set_title('Stimulation')
        
        axs[2].plot(t_emg, signalRect)
        axs[2].set_title("EMG rectifié")
        
        axs[3].plot(t_force, force)
        axs[3].set_title("Force (V)")
        
        axs[4].plot(t_force, forceCal)
        axs[4].set_title("Force calibrée (g)")
        
        
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
    
    
    #--------RAT 287-------------------
        # LOAD LE FICHIER pour une stimulation de 300mA
    dir_path = "C:/Users/Maxime/Desktop/FredericD/287/opto_gauche/exp1/"
    psth2 = P.Psth(dir_path, "opto") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
    psth2.loadDataFromDir(preFileNameChara = "_", postFileNameChara = "ma")
    psth2.calibrationSenseur = calibration.calibrationSenseur
    psth2.fromChannel2Psth(0.01, 0.025, 1, 2, OnePulsePerEvent = True) # fournir : (t_inf, t_supp, canal signal, canal événement)
    psth2.showAllPsth("") # fournir : ((rangée par, colonne de graphiques), le fichier où sera sauvegardé la figure ou rien(""))
    psth2.latenceVsEmg(True, False, True) # (first pulse by train, showplotlatence all stim, showplot result)
    