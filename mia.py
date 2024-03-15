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

def read_second_column(folder_path):
    channel_columns = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            if df.shape[1] >= 2:
                channel_columns.append(df.iloc[:, 1].tolist())
    return channel_columns

def plot_pca_variance(variance_explained):
    """
    Affiche un graphique de la variance expliquée par chaque composante principale d'une PCA.

    :param variance_explained: Une liste ou un tableau numpy contenant les pourcentages de variance expliquée par chaque composante.
    """
    composantes = np.arange(1, len(variance_explained) + 1)  # Composantes principales
    variance_cumulee = np.cumsum(variance_explained)  # Variance cumulée

    # Création du graphique 
    plt.figure(figsize=(10, 6))
    plt.bar(composantes, variance_explained, alpha=0.6, label='Variance Expliquée par Composante')
    plt.plot(composantes, variance_cumulee, color='red', marker='o', linestyle='dashed', 
             linewidth=2, markersize=12, label='Variance Cumulée')

    # Ajout des titres et labels
    plt.title('Variance Expliquée par les Composantes Principales (PCA)')
    plt.xlabel('Composantes Principales')
    plt.ylabel('Pourcentage de Variance Expliquée')
    plt.xticks(composantes)
    plt.legend()

    # Affichage du graphique
    plt.show()

def plot_psth(psth_results, dimension, temps_debut, incrementation,pca=False):
    psth_low, psth_mean, psth_high = psth_results
    indices = np.arange(temps_debut, temps_debut + len(psth_mean) * incrementation, incrementation)
    plt.plot(indices, psth_mean, color='black')
    
    # Dessiner les courbes pour les bornes inférieures et supérieures en mode invisible
    plt.plot(indices, psth_low, color='none')
    plt.plot(indices, psth_high, color='none')

    # Remplir l'espace entre les courbes de bornes inférieures et supérieures
    plt.fill_between(indices, psth_low, psth_high, color='red', alpha=0.5)

    # Dessiner à nouveau la courbe moyenne pour qu'elle soit bien visible
    plt.plot(indices, psth_mean, color='red')

    # Ajout des titres et des étiquettes
    
    plt.title(f'PSTH for PCA {dimension + 1}')
    plt.xlabel('Time')
    plt.ylabel('Signal intensity')

    # Afficher le graphique
    plt.show()

folder_path = "/Users/vincent/Downloads/Spike_per_time_bin/"
array = read_second_column(folder_path)
print(len(array[0]))
stim_list = [10000, 20000]
# plt.plot(np.arange(len(array[0])), array[0])
# plt.show()
# pca = PCA().fit(array)
# variance = pca.explained_variance_ratio_ * 100
# plot_pca_variance(variance)

reduction = 3
# # Réduire les données à X dimensions
pca = PCA(n_components=int(reduction))
data_transformed = pca.fit_transform(array)

data_cut = da.cut_individual_event(0.5,1,stim_list,data_transformed,0.02)
psth_results = da.PSTH(data_cut)

da.plot_psth(0.5,1,*psth_results,len(data_cut))