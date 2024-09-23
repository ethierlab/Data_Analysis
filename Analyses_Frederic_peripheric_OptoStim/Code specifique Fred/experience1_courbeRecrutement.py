import psth_class as P # classe qui contient toutes les fonctions nécessaire aux analyse 
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
        #axs[0].set_ylim(-limiteSup[0], limiteSup[0])
        axs[1].plot(t_emg, stimSeg)
        axs[1].set_title('Stimulation')
        #axs[1].set_ylim(0, limiteSup[1])
        axs[2].plot(t_emg, signalRect)
        axs[2].set_title("EMG rectifié")
        #axs[2].set_ylim(-0.5, limiteSup[2])
        axs[3].plot(t_force, force)
        axs[3].set_title("Force (V)")
        #axs[3].set_ylim(4.235, limiteSup[3])
        axs[4].plot(t_force, forceCal)
        axs[4].set_title("Force calibrée (g)")
        #axs[4].set_ylim(-0.05, limiteSup[4])
        
        plt.show()


if __name__ == "__main__":
  
    # 1 fichier à la fois :

    # Choix du bon capteur force 100g ou 500g :

    capteur = 500
    
    if capteur == 500:
        #--------CALIBRATION DU CAPTEUR FORCE-500Grammes---------------
        dir_path = "T:/Projects/optogenetic_periph/Calibration1/"
        calibration = P.Psth(dir_path, "elect") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
        calibration.loadLabchartFromDir()
        calibration.calibrationForceVoltage(2,[0, 50, 55, 60, 70, 90, 110, 130]) # basé sur calibration faite le 25 juin 24
        m = calibration.calibrationSenseur["slope"] # pente de la calibration force
        b = calibration.calibrationSenseur["intercept"] # ordonnée a l'origine de la calibration force
    
    elif capteur == 100:
        #--------CALIBRATION DU CAPTEUR FORCE-100Grammes---------------
        dir_path = "T:/Projects/optogenetic_periph/Calibration2"
        calibration = P.Psth(dir_path, "elect") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
        calibration.loadLabchartFromDir()
        calibration.calibrationForceVoltage(2,[0, 7.4, 8.6, 13.6, 23.6, 43.6, 63.6, 83.6, 103.6]) # basé sur calibration faite le 25 juin 24
        m = calibration.calibrationSenseur["slope"] # pente de la calibration force
        b = calibration.calibrationSenseur["intercept"] # ordonnée a l'origine de la calibration force
    


    dir_path = "T:/Projects/optogenetic_periph/324/opto_droit/exp1"
    psth1 = P.Psth(dir_path, "opto") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
    psth1.loadDataFromDir("_","ma") # fournir (préfixe, sufixe) entourant la valeure de stim introduite dans fichier !!! doit changer nom de fonction si
    # 1 seul fichier dans dossier (exp 2 et 3 non programmé)
    psth1.calibrationSenseur = calibration.calibrationSenseur
    psth1.fromChannel2Psth(0.005, 0.025, 1, 2, OnePulsePerEvent = True) # fournir : (t_inf, t_supp, canal signal, canal événement)
    psth1.showAllPsth("") # fournir : ((rangée par, colonne de graphiques), le fichier où sera sauvegardé la figure ou rien(""))
    psth1.peak2peak("")  # fournir : (le fichier où sera sauvegardé la figure ou rien(""))
    psth1.fromChannel2PsthRectEmg(0.005, 0.110, 1, 2, OnePulsePerEvent = True) # inclut la latence dans le calcul
    psth1.showAllPsth("")
    psth1.courbeRecrutement([0, 0.2], "EMG rectifie","") # Fait les calculs sur l'étendu fourni au 1er argument
    psth1.fromChannel2PsthForce(0.005, 0.110, 3, 2, OnePulsePerEvent = True)
    psth1.showAllPsth("")
    psth1.courbeRecrutement([0, 0.2], "Force","") # Fait les calculs sur l'étendu fourni au 1er argument
    psth1.latenceVsEmg(True, False, True) # (first pulse by train, showplotlatence all stim, showplot result)

    
    