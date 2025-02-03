# Sortie des données pour Prism

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
import pickle as pk
import pandas as pd




def exportLabChartChannelToExcel(path, preFileNameChara = "_", postFileNameChara = "ma"):
        """ Read the 'adi' file in the folder. 1 file, 1 recording, 1 condition, 1 signal channel and 1 event channel(stim)
            Extract the stimulation curent amplitude value in the file name, base on pre and post File character "_". More than one file
            each file only one block 
        """
            ## # All id numbering is 1 based, first channel, first block
            # When indexing in Python we need to shift by 1 for 0 based indexing
            # Functions however respect the 1 based notation ...
            
        if not os.path.exists(path):
            print(f"Error: The directory '{path}' does not exist.")
            return
        # List all the files in the given directory
        files = os.listdir(path)
        n_files = len(files)
        
        
        # condition variable indépendante : intensité ma
        courant_val = []
        adi_out = []
        freq = 10000

        
        if n_files == 0:
            print("Error no files to analyse")
        elif n_files == 1:
            if files[0].endswith(".adicht"):
                txt_path = os.path.join(path, files[0])
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
                    txt_path = os.path.join(path, file_name)
                    ind1 = file_name.rfind(preFileNameChara)
                    ind2 = file_name.rindex(postFileNameChara)
                    courant_val.append(float(file_name[ind1+1:ind2]))
                    f = adi.read_file(txt_path)
                    adi_out.append(f)
                else:
                    print("Error at least one file is not a labchart one")
                
        
        # construction du dataframe panda :
        u=0
        df =  pd.DataFrame()
        for enr in adi_out:
            signal = enr.channels[0].get_data(1)
            evenement = enr.channels[1].get_data(1)
            n = len(signal)
            listeCourant = np.ones(n)*courant_val[u]
            time = np.arange(n)
            time  = time/freq
            listParametre = list(zip(listeCourant, time, signal, evenement))
            df = pd.concat([df, pd.DataFrame(listParametre, columns = ["Courant", "Temps", "Signal", "Evenement"])])
            u+=1
            
        df.to_excel('C:/Users/Maxime/Desktop/test/dataExcel.xlsx')       
       
def exportPsthToExcel(self):
    "Permet d'avoir les données pour faire un histogramme"
    #  self.t_inf = t_inf
    #  self.t_supp = t_supp
    #  self.psth_compil = psth_compil
    #  self.min_max = min_max
    time = np.arange(len(self.psth_compil[0]))
    n = len(self.psth_compil[0])
    time  = time/self.freq
    time -= self.t_inf
    
    # for pst in self.psth_compil:
    #     plt.plot(time, pst)
    #     plt.show()

     # construction du dataframe panda :
    u=0
    df =  pd.DataFrame()
    for pst in self.psth_compil:
        
        listeCourant = np.ones(n)*self.courant_val[u]
        listParametre = list(zip(listeCourant, time, self.psth_compil_min[u], pst, self.psth_compil_max[u]))
        df = pd.concat([df, pd.DataFrame(listParametre, columns = ["Courant", "Temps", "Psth EMG Min", "Psth EMG", "Psth EMG Max"])])
        u+=1
        
    df.to_excel('C:/Users/Maxime/Desktop/test/PsthExcel.xlsx')
    return time

def exportTousSegmentPsth(self):
    "Exporte tous les segments qui ont servis à produire les psths"
    self.dataFrameSegmentPsth.to_excel('C:/Users/Maxime/Desktop/test/PsthSegmentExcel.xlsx')


def exportCourbRecruToExcel(self):
    "Calcul et exporte les points pour la courbe de recrutement vers excel"
    # Plusieurs façons possibles de calculer l'amplitude de l'emg :

    # Peak to peak à partir de chacune des stimulations
    
    # construction du dataframe panda :
    
    df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame(self.etenduTouSegment, columns = ["Courant", "Repetition", "Amplitude Max Emg"])])
    df.to_excel('C:/Users/Maxime/Desktop/test/CourbeRecrutementExcel.xlsx')

def exportLatence(self):
    "Exporte les latences de chacun des emgs" # Attention l'algorythme fait des erreurs

    listParametre = list(zip(self.indice_latence, self.val_courant, self.emg_amplitude, self.niemeRep))
    df = pd.DataFrame(listParametre, columns = ["Latence", "Courant", "Emg amplitude", "Repetition #"])
    df.to_excel('C:/Users/Maxime/Desktop/test/latence.xlsx')   
    

 
if __name__ == "__main__":
  
    # 1 fichier à la fois :

    # Choix du bon capteur force 100g ou 500g :

    capteur = 500
    
    if capteur == 500:
        #--------CALIBRATION DU CAPTEUR FORCE-500Grammes---------------
        dir_path = "C:/Users/Maxime/Desktop/FredericD/Calibration"
        calibration = P.Psth(dir_path, "elect") # fournir (dossier chemin ou "", type de stim : "elect" ou "elec" )
        calibration.loadLabchartFromDir()
        calibration.calibrationForceVoltage(2,[0, 50, 55, 60, 70, 90, 110, 130]) # basé sur calibration faite le 25 juin 24
        m = calibration.calibrationSenseur["slope"] # pente de la calibration force
        b = calibration.calibrationSenseur["intercept"] # ordonnée a l'origine de la calibration force
    
    elif capteur == 100:
        #--------CALIBRATION DU CAPTEUR FORCE-100Grammes---------------
        dir_path = "C:/Users/Maxime/Desktop/FredericD/Calibration2"
        calibration = P.Psth(dir_path, "elect") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
        calibration.loadLabchartFromDir()
        calibration.calibrationForceVoltage(2,[0, 7.4, 8.6, 13.6, 23.6, 43.6, 63.6, 83.6, 103.6]) # basé sur calibration faite le 25 juin 24
        m = calibration.calibrationSenseur["slope"] # pente de la calibration force
        b = calibration.calibrationSenseur["intercept"] # ordonnée a l'origine de la calibration force


    dir_path = "Z:/Projects/optogenetic_periph/255/opto_gauche/exp1"
    dir_path = "C:/Users/Maxime/Desktop/FredericD/255/gauche"
    dir_path = "Z:/Projects/optogenetic_periph/230/opto_droit/exp1"
    #exportLabChartChannelToExcel(dir_path) 
    
    ratOpto = P.Psth(dir_path, "opto") # fournir (dossier chemin ou "", type de stim : "elect" ou "opto" )
    ratOpto.loadDataFromDir("_","ma") # fournir (préfixe, sufixe) entourant la valeure de stim introduite dans fichier !!! doit changer nom de fonction si
    # 1 seul fichier dans dossier (exp 2 et 3 non programmé)
    ratOpto.calibrationSenseur = calibration.calibrationSenseur
    ratOpto.fromChannel2Psth(0.005, 0.025, 1, 2, OnePulsePerEvent = True) # fournir : (t_inf, t_supp, canal signal, canal événement)
    ratOpto.psth_time = exportPsthToExcel(ratOpto)
    ratOpto.peak2peak("")  # fournir : (le fichier où sera sauvegardé la figure ou rien(""))
    exportTousSegmentPsth(ratOpto)
    exportCourbRecruToExcel(ratOpto)
    ratOpto.latenceVsEmg(True, False , False) # (first pulse by train, showplotlatence all stim, showplot result)
    # Attention pas toujours bon la latence vaut mieux la faire visuellement
    

    ratOpto.latence(True, 50, 1, 4)
    exportLatence(ratOpto)
