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
    # experience 3 : 10 mins, 5 hz, env. 30% de l'emg max, electrique et optogenetique doivent générés un emg équivalent
    

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
    
<<<<<<< HEAD
    dir_path = "Z:/Projects/optogenetic_periph/255/opto_gauche/exp3"
=======

    
    dir_path = "T:/Projects/optogenetic_periph/330/opto_droit/exp3"
>>>>>>> 2c3b4f518e0574d7b4175577fc718c16d4779ae9
    fatigue = P.Psth(dir_path, "opto") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
    fatigue.loadLabchartFromDir() # fournir (préfixe, sufixe) entourant la valeure de stim introduite dans fichier !!! doit changer nom de fonction si
    fatigue.calibrationSenseur = calibration.calibrationSenseur # calibre la force 
    fatigue.fromChannel2PsthIntraTrainExp3(1, 2, 5, 15, 30, True)

<<<<<<< HEAD
    # dir_path = "C:/Users/Maxime/Desktop/FredericD/291/electrique/exp2"
    # fatigue = P.Psth(dir_path, "elec") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
    # fatigue.loadLabchartFromDir() # fournir (préfixe, sufixe) entourant la valeure de stim introduite dans fichier !!! doit changer nom de fonction si
    # fatigue.calibrationSenseur = calibration.calibrationSenseur # calibre la force 
    # fatigue.fromChannel2PsthIntraTrainExp3(1, 2, 5, 15, 30, True)
   
=======
    dir_path = "T:/Projects/optogenetic_periph/330/elec_droit/exp3"
    fatigue = P.Psth(dir_path, "elec") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
    fatigue.loadLabchartFromDir() # fournir (préfixe, sufixe) entourant la valeure de stim introduite dans fichier !!! doit changer nom de fonction si
    fatigue.calibrationSenseur = calibration.calibrationSenseur # calibre la force 
    fatigue.fromChannel2PsthIntraTrainExp3(1, 2, 5, 15, 30, True)


    
>>>>>>> 2c3b4f518e0574d7b4175577fc718c16d4779ae9
