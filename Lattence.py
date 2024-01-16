import data_analysis as da
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path_donnees= "/Users/freddydagenais/Desktop/Maitrise/code/230/opto_peripherique /230_stim_opto_periph_droit_csv/psth"
# path_saving =


def find_latency(data, window_size=10, k=0.08):
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
    df = pd.read_csv(file_path, skiprows=7)
    # Initialize an empty dictionary to store arrays
    arrays_dict = {}
    # Iterate over each column
    for column in df.columns:
        # Convert column to numpy array and store in dictionary
        arrays_dict[column] = df[column].to_numpy()
    return arrays_dict

run = 0 
amplitude = []
courant = []
files = os.listdir(path_donnees)
latence = []
for file_name in files:
    if file_name.endswith(".csv"):
        
        PATH = path_donnees + "/" + file_name
        # print(PATH) 
        data = csv_to_np_arrays(PATH)
        before = len(amplitude)
        # amplitude.append(abs(np.max(data["Average"])- np.min(data["Average"])))
        # amplitude = amplitude[amplitude>]
        latence.append(find_latency(data['Average']))
        # print(latence)
        amplitude.append(abs(np.max(data["Average"])- np.min(data["Average"])))

        
        plt.scatter(data["Time (s)"],data["Average"])
        plt.axvline(x=(latence[run])/da.find_frequency(data["Time (s)"]), color='r', linestyle='--')
        run +=1
        plt.show()

        plt.scatter(amplitude,latence)
        plt.ylabel("Lattence (s)")
        plt.xlabel("Amplitude (mV)")
        plt.show()


# print(latence)
# print(amplitude)



plt.scatter(amplitude,latence)
plt.ylabel("Lattence (s)")
plt.xlabel("Amplitude (mV)")
plt.savefig("Courbe de lattence selon l'amplitude.svg", dpi=700)
plt.show()
