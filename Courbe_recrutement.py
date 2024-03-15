import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit



# path_donnees= "/Users/freddydagenais/Desktop/Maitrise/code/255/exp1/csv/psth"
# path_saving = "/Users/freddydagenais/Desktop/Maitrise/code/255/exp1/csv/psth/courbe_recrutement" # Chemin ou tu veux save tes données
path_donnees = "/Users/vincent/Desktop/fred/230/Droit/psth"
path_saving = "/Users/vincent/Desktop/fred/230/Droit" # Chemin ou tu veux save tes données
path_donnees_2 = "/Users/vincent/Desktop/fred/235/PSTH"
path_donnees_3 = "/Users/vincent/Desktop/fred/255/psth"


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

amplitude = [[],[],[]]
courant = [[],[],[]]

# print(files)
liste_path = [path_donnees, path_donnees_2, path_donnees_3]
for i, path in enumerate(liste_path):
    files = os.listdir(path)
    files = sorted(files)
    # print(files)
    for file_name in files:
        if file_name.endswith(".csv"):
            # print(file_name)
            PATH = path + "/" + file_name
            prem_instance = PATH.rfind('_')
            if prem_instance != -1:
                index_back = PATH[:prem_instance].rfind('_')
            if i == 0:
                courant[i].append(int((PATH[index_back+1: prem_instance])))
            else:
                courant[i].append(int((PATH[index_back+1: prem_instance-2])))
            data = csv_to_np_arrays(PATH)
        
            amplitude[i].append(abs(np.max(data["Average"])- np.min(data["Average"])))
    # normalisation = amplitude[i]/max(amplitude[i])
# global_max = max(max(sublist) for sublist in amplitude if sublist)
# normalisation = [[value / global_max for value in sublist] for sublist in amplitude]
# print(normalisation_courant)
# print(normalisation)
normalisation_courant = []
normalisation = []
for serie in courant:
    min_val = min(serie)
    max_val = max(serie)
    serie_normalisee = [(x - min_val) / (max_val - min_val) if max_val != min_val else 0 for x in serie]
    normalisation_courant.append(serie_normalisee)
#Curve fitting
for serie in amplitude:
    min_val = min(serie)
    max_val = max(serie)
    serie_normalisee = [(x - min_val) / (max_val - min_val) if max_val != min_val else 0 for x in serie]
    normalisation.append(serie_normalisee)
for i in range(len(normalisation_courant)):
    courant_actuelle = normalisation_courant[i]
    normalisation_actuelle = normalisation[i]
    
    # Ajustement de la courbe
    params, covariance = curve_fit(courbe, courant_actuelle, normalisation_actuelle, p0=[0.7, 200, 600])
    
    # Création de données de courbe ajustée pour le tracé
    x_fit = np.linspace(min(courant_actuelle), max(courant_actuelle), 100)
    y_fit = courbe(x_fit, *params)
    plt.plot(x_fit, y_fit)
for i in range(3):
    plt.scatter(normalisation_courant[i], normalisation[i], label = f"Courbe {i+1}")
plt.xlabel("Courant %")
plt.ylabel("Amplitude normalisé(mV)")
plt.legend()
plt.savefig('Courbe Recrutement funky.svg', dpi=700)
plt.show()
