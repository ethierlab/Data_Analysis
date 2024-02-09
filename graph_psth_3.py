import os
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import csv
import numpy as np
import umap
import data_analysis as da
import pandas as pd


def csv_to_numpy(file_path, encoding='utf-8'):
    if not os.path.exists(file_path):
        print(f"Le fichier ou le dossier {file_path} n'existe pas.")
        return None
    data_list = []

    def process_file(file_path, encoding='utf-8'):
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader, None)  # Skip the header if it exists
                for row in csv_reader:
                    data_list.append(row)
        except UnicodeDecodeError as e:
            print(f"Erreur de décodage pour le fichier {file_path}: {e}. Essayez un encodage différent.")

    def process_directory(directory_path, encoding='utf-8'):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                process_file(item_path, encoding)
            elif os.path.isdir(item_path):
                process_directory(item_path, encoding)  # Recursive call

    if os.path.isfile(file_path):
        process_file(file_path, encoding)
    elif os.path.isdir(file_path):
        process_directory(file_path, encoding)
    
    # Transpose the list of rows to get a list of columns
    transposed_data = list(zip(*data_list))

    # Convert each column to a separate NumPy array
    numpy_columns = []
    for column in transposed_data:
        try:
            float_column = np.array(column, dtype=float)
            numpy_columns.append(float_column)

        except ValueError:
            numpy_columns.append(np.array(column))
    # Combiner les colonnes en un seul tableau numpy        
    combined_array = np.column_stack(numpy_columns)
    return combined_array

def read_csv_and_transform(file_path, base_change_matrix):
    # Lire le fichier CSV et le convertir en matrice NumPy
    df = pd.read_csv(file_path)
    data_matrix = df.values

    # Vérifier que les dimensions sont compatibles pour la multiplication
    if data_matrix.shape[1] == base_change_matrix.shape[0]:
        # Multiplier la matrice de données par la matrice de changement de base
        transformed_matrix = np.dot(data_matrix, base_change_matrix)
        return transformed_matrix
    else:
        print("Les dimensions des matrices ne sont pas compatibles pour la multiplication.")
        return None

def process_folder_for_psth(folder_path, column_index, base_change_matrix=None):
    all_column_data = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            transformed_data = read_csv_and_transform(file_path, base_change_matrix)
            if transformed_data is not None:
                all_column_data.append(transformed_data[:, column_index])

    # Trouver la longueur de la liste la plus courte
    min_length = min(map(len, all_column_data))

    # Tronquer toutes les listes à la longueur de la liste la plus courte
    truncated_data = [col[:min_length] for col in all_column_data]

    return truncated_data

def process_csv_files_in_folder(folder_path, column_index):
    all_column_data = []

    # Lire chaque fichier CSV et extraire la colonne spécifiée
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            data_matrix = df.values
            all_column_data.append(data_matrix[:, column_index])

    # Trouver la longueur de la liste la plus courte
    min_length = min(map(len, all_column_data))

    # Tronquer toutes les listes à la longueur de la liste la plus courte
    truncated_data = [col[:min_length] for col in all_column_data]

    return truncated_data


def plot_psth(psth_results, dimension, temps_debut, incrementation,pca=False):
    psth_low, psth_mean, psth_high = psth_results
    indices = np.arange(temps_debut, temps_debut + len(psth_mean) * incrementation, incrementation)
    plt.plot(indices, psth_mean, color='black')
    
    # Dessiner les courbes pour les bornes inférieures et supérieures en mode invisible
    plt.plot(indices, psth_low, color='none')
    plt.plot(indices, psth_high, color='none')

    # Remplir l'espace entre les courbes de bornes inférieures et supérieures
    plt.fill_between(indices, psth_low, psth_high, color='red', alpha=0.5, label='last 3')

    # Dessiner à nouveau la courbe moyenne pour qu'elle soit bien visible
    plt.plot(indices, psth_mean, color='red')

    # Ajout des titres et des étiquettes


def plot_psth_g(psth_results, dimension, temps_debut, incrementation,pca=False):
    psth_low, psth_mean, psth_high = psth_results
    indices = np.arange(temps_debut, temps_debut + len(psth_mean) * incrementation, incrementation)
    plt.plot(indices, psth_mean, color='black')
    
    # Dessiner les courbes pour les bornes inférieures et supérieures en mode invisible
    plt.plot(indices, psth_low, color='none')
    plt.plot(indices, psth_high, color='none')

    # Remplir l'espace entre les courbes de bornes inférieures et supérieures
    plt.fill_between(indices, psth_low, psth_high, color='green', alpha=0.5, label='first 3')

    # Dessiner à nouveau la courbe moyenne pour qu'elle soit bien visible
    plt.plot(indices, psth_mean, color='green')
    # Afficher le graphique
folder_path = '/Users/vincent/Desktop/data Michael/mai-9-1/3first' 
folder_path_1= "/Users/vincent/Desktop/data Michael/mai-9-1/3last"
array = csv_to_numpy(folder_path)
array_1 = csv_to_numpy(folder_path_1)
temps_debut = - 1500/1000
incrementation = 50/1000

print(min(array.shape[1], array_1.shape[1]))

for neurone in range(min(array.shape[1], array_1.shape[1])):
    
    first_3 = process_csv_files_in_folder(folder_path, neurone)
    last_3 = process_csv_files_in_folder(folder_path_1, neurone)
    psth_3first = da.PSTH(first_3)
    psth_3last = da.PSTH(last_3)
    plt.title(f'PSTH for neuron {neurone + 1}')
    
    plt.xlabel('time')
    plt.ylabel('Value')
    plot_psth_g(psth_3first,neurone, temps_debut, incrementation)
    plot_psth(psth_3last,neurone, temps_debut, incrementation)
    plt.legend()
    plt.savefig(f'psth_{neurone}.png')
    plt.clf()
    # plt.show()