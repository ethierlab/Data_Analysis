import data_analysis as da
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
chan_events = "2" # Enter the channel name for the Events
Signal_psth = "1"  # Enter the channel name for the PSTH
t_inf = 0.005 # (seconds) Enter time value before stimulation
t_supp = 0.015 # (seconds) Enter time value after stimulation
path_donnees= "/Users/freddydagenais/Desktop/Maitrise/code/255/exp1/csv"
path_saving = "/Users/freddydagenais/Desktop/Maitrise/code/255/exp1/csv/psth" # Chemin ou tu veux save tes données

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

        print(time)
        freq = find_frequency(time)
        
        #Plot les graph pour Y sig t les stims
        # plt.plot(time,y_sig)
        # plt.plot(time,events)
        # plt.show()
        index = find_event_index(events)
        sample = cut_individual_event(t_inf, t_supp, index, y_sig, freq)
        plot_psth(t_inf,t_supp,*da.PSTH(sample), len(sample), PATH[PATH.rfind("/")+1:-4])
        plt.savefig(f'{file_name[:-4]}.svg',dpi=700)
        plt.show()
        data = da.generate_PSTH_data(sample, t_inf, t_supp)
        filename = PATH[PATH.rfind("/")+1:-4] + "_PSTH.csv"
        da.create_PSTH_CSV(PATH, path_saving+'/'+ filename, data)
