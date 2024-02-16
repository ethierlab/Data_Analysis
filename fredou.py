import data_analysis as da
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

chan_events = "3" # Enter the channel name for the Events
Signal_psth = "1"  # Enter the channel name for the PSTH
t_inf = 0.000 # (seconds) Enter time value before stimulation
t_supp = 0.01 # (seconds) Enter time value after stimulation
path_donnees= "/Users/vincent/Desktop/fred/235"
path_saving = "/Users/vincent/Desktop/fred/243" # Chemin ou tu veux save tes données

def csv_to_np_arrays(file_path):
    """
    Reads a CSV file and converts each column into a separate NumPy array.
    Parameters:
    - file_path (str): The path to the CSV file to be read.
    The function reads the CSV file, skipping the first 7 rows, and iterates through each column,
    converting it into a NumPy array. The resulting arrays are stored in a dictionary where the
    keys are the column names, and the values are the corresponding NumPy arrays.
    Note: The function assumes that the first 7 rows of the CSV file do not contain relevant data
    and skips them. Make sure that this assumption aligns with the structure of your CSV file.
    Returns:
    - arrays_dict (dict): A dictionary where keys are the column names, and values are the corresponding
      NumPy arrays.
    """
    # Read the CSV file with pandas, skipping the first 7 rows
    df = pd.read_csv(file_path)
    # Initialize an empty dictionary to store arrays
    arrays_dict = {}
    # Iterate over each column
    for column in df.columns:
        # Convert column to numpy array and store in dictionary
        arrays_dict[column] = df[column].astype(float).to_numpy()
    return arrays_dict
def find_frequency(raw_time):
    '''
    Find the sampling frequency
    Parameters:
    raw_time (list): list of time
    Return:
    An integer that represent the sampling frequency
    '''
    return int(1/(float(raw_time[1]) - float(raw_time[0])))
def find_event_index(signal:list):
    """
    Find the index of peaks and adds them to a list
    Parameters:
    signal (list): list of Signal that contains the peaks to find
    Return: List of index where peaks were found
    """

    for i,elem in enumerate(signal):
        if elem > 0.1:
            signal[i] = 1
        else:
            signal[i] = 0
    diff = np.diff(signal)
    return np.where(diff >= 0.9)[0]+ 1

def cut_individual_event(t_inf, t_supp, peak_index:list, opto_stim:list, frequency):
    """
    Cut the values of given list in 
    
    Parameters:
    t_inf (int): time before the spike 
    t_supp (int): time after the spike
    peak_index (list): List of the peak indices
    brain_stim (list): List of stimulation signal values

    Return: list of list of cut values from lower bound to upper bound
    """
    stim_cut = [] # list for the brain stimulation data separated for each spike
    inf_offset =  int(t_inf * frequency)
    supp_offset = int(t_supp * frequency)

    for val in peak_index:
        if val - inf_offset < 0 and val == min(peak_index):
            raise ValueError("Warning! The time before the events is too large. Change the t_inf parameter.")

        if val + supp_offset + 1 > len(opto_stim):
            raise ValueError("Warning! The time after the events is too large. Change the t_supp parameter.")

        stim_cut.append(opto_stim[val - inf_offset : val + supp_offset + 1])

    return stim_cut

def plot_psth(t_inf, t_supp, psth_low, psth_mean, psth_high, nb_samp, title):
    time = np.linspace(-t_inf, t_supp, len(psth_mean))

    plt.plot(time, psth_mean, color='black')
    
    # Dessiner les deux premières courbes invisibles
    plt.plot(time, psth_low, color='none')
    plt.plot(time, psth_high, color='none')

    # Remplir l'espace entre les deux courbes invisibles
    plt.fill_between(time, psth_low, psth_high, color='red', alpha=0.5)

    # Dessiner la troisième courbe visible
    plt.plot(time, psth_mean, color='red')

    # Afficher le graphique
    plt.title(f'PSTH de {title} avec {nb_samp}')
    plt.xlabel('Time [s]')
    plt.ylabel('Signal Intensity')
    
def peak_to_peak(signal):
    return max(signal) - min(signal)

def cut_signal_and_time(time, signal, first_cut, second_cut, time_cut_after_second):
    # Trouve l'index de départ pour la première coupure
    start_index = np.argmin(np.abs(time - first_cut))
    
    # Trouve l'index pour la deuxième coupure
    end_index = np.argmin(np.abs(time - second_cut))
    
    # Calcule l'index de fin basé sur time_cut_after_second
    if time_cut_after_second > 0:
        # Calcule l'interval moyen entre les points de temps
        average_interval = np.mean(np.diff(time))
        number_of_intervals = int(time_cut_after_second / average_interval)
        final_end_index = min(end_index + number_of_intervals, len(signal))
    else:
        final_end_index = end_index
    
    # Calcule l'index de fin basé sur time_cut_after_second
    average_interval = sum([time[i+1] - time[i] for i in range(len(time)-1)]) / (len(time) - 1)
    number_of_intervals = int(time_cut_after_second / average_interval)
    final_end_index = min(end_index + number_of_intervals, len(signal))
    
    # Retourne les portions nécessaires
    time_first_cut = time[start_index:end_index+1]
    signal_first_cut = signal[start_index:end_index+1]
    time_second_cut = time[end_index+1:final_end_index]
    signal_second_cut = signal[end_index+1:final_end_index]
    
    return time_first_cut, signal_first_cut, time_second_cut, signal_second_cut


files = os.listdir(path_donnees)
files = sorted(files)
print(files)
for file_name in files:
    if file_name.endswith(".csv"):
        PATH = path_donnees + "/" + file_name
        dictio = csv_to_np_arrays(PATH)
        keys_list = []
        # Add keys from the dictionary to the list
        for key in dictio.keys():
            keys_list.append(key)
            try:
                time =  dictio[keys_list[0]]
                events = dictio[chan_events]
                y_sig = dictio[Signal_psth]
            except KeyError:
                print("The channel name you entered you entered does not exist")
                
        # plt.plot(time, y_sig)
        # plt.plot(time, events)
        # plt.show()
        # print(time)
        # print(events)
        # print(time)
        freq = find_frequency(time)
        
        #Plot les graph pour Y sig t les stims
        # plt.plot(time,y_sig)
        # plt.plot(time,events)
        # plt.show()
        index = find_event_index(events)
        # print(index)
        sample = cut_individual_event(t_inf, t_supp, index, y_sig, freq)
        # print(sample)
        # print(da.PSTH(sample))
        plot_psth(t_inf,t_supp,*da.PSTH(sample), len(sample), PATH[PATH.rfind("/")+1:-4])
        # plt.savefig(f'{file_name[:-4]}.png',dpi=700)
        plt.show()

        time = np.linspace(-t_inf, t_supp, len(da.PSTH(sample)[0]))
        action = input("Voulez-vous couper le signal? (y/n): ")
        if action == 'y':  
            coupure_1 =  input("A quel temps veux-tu la première coupure")## AJOUTER LE TEMPS OU TU VEUX COUPER
            coupure_2 = input("A quel temps veux-tu la deuxième coupure")## AJOUTER LE TEMPS OU TU VEUX COUPER
            temps_coupure =  input("Combien de temps veux-tu après la deuxième coupure")
        
            A_time, A_signal, B_time, B_signal = cut_signal_and_time(time, da.PSTH(sample)[1], coupure_1, coupure_2, temps_coupure)
            plt.plot(A_time, A_signal, color='black')
            plt.title(f'Signal M pour {file_name[:-4]}')
            plt.xlabel('Time [s]')
            plt.ylabel('Signal Intensity')
            plt.show()
        
            plt.plot(B_time, B_signal, color='black')
            plt.title(f'Signal H pour {file_name[:-4]}')
            plt.xlabel('Time [s]')
            plt.ylabel('Signal Intensity')
            plt.show()
            print(f'{file_name[:-4]} ')
            print(f'le peak to peak du signal M est de:{peak_to_peak(A_signal)}')
            print(f'le peak to peak du signal H est de:{peak_to_peak(B_signal)}')
        # data = da.generate_PSTH_data(sample, t_inf, t_supp)
        # filename = PATH[PATH.rfind("/")+1:-4] + "_PSTH.csv"
        # da.create_PSTH_CSV(PATH, path_saving+'/'+ filename, data)





