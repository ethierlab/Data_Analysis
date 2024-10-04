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

if __name__ == "__main__":
  
    # 1 fichier à la fois :

    # Choix du bon capteur force 100g ou 500g :

    capteur = 100
    
    if capteur == 500:
        #--------CALIBRATION DU CAPTEUR FORCE-500Grammes---------------
        dir_path = "Z:/Projects/optogenetic_periph/Calibration1/"
        calibration = P.Psth(dir_path, "elect") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
        calibration.loadLabchartFromDir()
        calibration.calibrationForceVoltage(2,[0, 50, 55, 60, 70, 90, 110, 130]) # basé sur calibration faite le 25 juin 24
        m = calibration.calibrationSenseur["slope"] # pente de la calibration force
        b = calibration.calibrationSenseur["intercept"] # ordonnée a l'origine de la calibration force
    
    elif capteur == 100:
        #--------CALIBRATION DU CAPTEUR FORCE-100Grammes---------------
        dir_path = "Z:/Projects/optogenetic_periph/Calibration2"
        calibration = P.Psth(dir_path, "elect") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
        calibration.loadLabchartFromDir()
        calibration.calibrationForceVoltage(2,[0, 7.4, 8.6, 13.6, 23.6, 43.6, 63.6, 83.6, 103.6]) # basé sur calibration faite le 25 juin 24
        m = calibration.calibrationSenseur["slope"] # pente de la calibration force
        b = calibration.calibrationSenseur["intercept"] # ordonnée a l'origine de la calibration force
    


    dir_path = "Z:/Projects/optogenetic_periph/324/opto_droit/exp4"
    fatigueFreq = P.Psth(dir_path, "opto") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
    fatigueFreq.loadDataFromDir("_","hz") # fournir (préfixe, sufixe) entourant la valeure de stim introduite dans fichier !!! doit changer nom de fonction si
    fatigueFreq.calibrationSenseur = calibration.calibrationSenseur # calibre la force 

    fatigueFreq.fromChannel2PsthIntraTrain(0.005, 0.025, 1, 2, plot = False)
    
    
    # psth1.fromChannel2Psth(0.005, 0.025, 1, 2, OnePulsePerEvent = True) # fournir : (t_inf, t_supp, canal signal, canal événement)
    # psth1.showAllPsth("") # fournir : ((rangée par, colonne de graphiques), le fichier où sera sauvegardé la figure ou rien(""))
    # psth1.peak2peak("")  # fournir : (le fichier où sera sauvegardé la figure ou rien(""))
    # psth1.fromChannel2PsthRectEmg(0.005, 0.110, 1, 2, OnePulsePerEvent = True) # inclut la latence dans le calcul
    # psth1.showAllPsth("")
    # psth1.courbeRecrutement([0, 0.2], "EMG rectifie","") # Fait les calculs sur l'étendu fourni au 1er argument
    # psth1.fromChannel2PsthForce(0.005, 0.110, 3, 2, OnePulsePerEvent = True)
    # psth1.showAllPsth("")
    # psth1.courbeRecrutement([0, 0.2], "Force","") # Fait les calculs sur l'étendu fourni au 1er argument
    # psth1.latenceVsEmg(True, True, True) # (first pulse by train, showplotlatence all stim, showplot result)

    
    