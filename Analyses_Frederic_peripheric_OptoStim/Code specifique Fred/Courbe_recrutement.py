import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit



path_donnees= "/Users/freddydagenais/Desktop/Maitrise/code/255/exp1/csv/psth"
path_saving = "/Users/freddydagenais/Desktop/Maitrise/code/255/exp1/csv/psth/courbe_recrutement" # Chemin ou tu veux save tes données

def courbe(x, a, b, c):
    return a / (1 + np.exp((-1 * (x - c))/b))

# def courbe(x, a, b, c):
#     return a*x +b

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

amplitude = []
courant = []
files = os.listdir(path_donnees)
files = sorted(files)
# print(files)

for file_name in files:
    if file_name.endswith(".csv"):
        PATH = path_donnees + "/" + file_name
        prem_instance = PATH.rfind('_')
        if prem_instance != -1:
            index_back = PATH[:prem_instance].rfind('_')
        
        courant.append(int((PATH[index_back+1: prem_instance-2])))
        data = csv_to_np_arrays(PATH)
    
        amplitude.append(abs(np.max(data["Average"])- np.min(data["Average"])))

#Curve fitting
params, covariance = curve_fit(courbe, courant, amplitude, p0=(0.7,200, 600))
print(params)

x_fit = np.linspace(min(courant), max(courant),100)

y_fit = courbe(x_fit, *params)

plt.plot(x_fit, y_fit)
plt.scatter(courant, amplitude)
plt.xlabel("Courant (mA)")
plt.ylabel("Amplitude (mV)")
plt.savefig('Courbe Recrutement 230 Droit.svg', dpi=700)
plt.show()
