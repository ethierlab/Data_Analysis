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

def cut_individual_event(t_inf, t_supp, peak_index:list, brain_stim:list, frequency):
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
    inf_offset =  int(t_inf / frequency)
    supp_offset = int(t_supp / frequency)

    for val in peak_index:
        if val - inf_offset < 0 and val == min(peak_index):
            raise ValueError("Warning! The time before the events is too large. Change the t_inf parameter.")
        
        if val + supp_offset + 1 > len(brain_stim):
            raise ValueError("Warning! The time after the events is too large. Change the t_supp parameter.")

        stim_cut.append(brain_stim[val - inf_offset : val + supp_offset + 1])

    return stim_cut

def read_column(folder_path, stim_path):
    channel_columns = []
    for filename in os.listdir(folder_path):
        print(filename)
        if filename == stim_path.split('/')[-1]:
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            if not df.empty:
                stim_column = df.iloc[:, 0].tolist()
        elif filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            if df.shape[1] >= 2:
                channel_columns.append(df.iloc[:, 1].tolist())
    return channel_columns, stim_column

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

def map_events_to_time_bins(event_times, time_bins):
    
    absolute_diffs_matrix = np.abs(np.subtract.outer(event_times, time_bins))
    closest_indices = np.argmin(absolute_diffs_matrix, axis=1)
    
    return closest_indices.tolist()

def plot_psth(t_inf, t_supp, psth_low, psth_mean, psth_high, nb_samp):
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
    plt.title(f'PSTH with n = {nb_samp} samples')
    plt.xlabel('Time [s]')
    plt.ylabel('Signal Intensity')
    

folder_path = "/Users/vincent/Desktop/Mia/219_dataset_50ms"
stim_path = "/Users/vincent/Desktop/Mia/219_dataset_50ms/219_Ane_Stim_t.csv"
array, stim_list = read_column(folder_path,stim_path)
step_size = 0.05
total_duration = (len(array[0]) - 1) * step_size
for elem in array:
    print(len(elem))

array = np.array(array)
# print(np.arange(0, total_duration + step_size, step_size))
stim_list = map_events_to_time_bins(stim_list[1:], np.arange(0, total_duration + step_size, step_size))
# print(stim_list*step_size)
print(len(stim_list))
# plt.plot(np.arange(len(array[0])), array[0])
# plt.show()
pca = PCA().fit(array)
variance = pca.explained_variance_ratio_ * 100
plot_pca_variance(variance)

reduction = 3
# # Réduire les données à X dimensions
pca = PCA(n_components=int(reduction))
data_transformed = pca.fit_transform(array.T)
print(data_transformed.shape)

data_cut = cut_individual_event(1,2,stim_list,data_transformed,step_size)
data_cut = np.array(data_cut)
print(data_cut.shape)
psth = []
for i in range(data_cut.shape[2]):
    slice_2d = data_cut[:, :, i]
    psth_result = da.PSTH(slice_2d)
    psth.append(psth_result)
print(psth_result)
premier_point_couleur = 'green'
point_specifique_couleur = 'blue'
dernier_point_couleur = 'red'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(psth[0][1], psth[1][1], psth[2][1], label='Mean Trajectory')
mean_trajectory = np.array([psth[0][1], psth[1][1], psth[2][1]])  # Une matrice 3 x N des points de la trajectoire moyenne

indice_point_specifique = 19  # Indice du point spécifique

ax.scatter(mean_trajectory[0,0], mean_trajectory[1,0], mean_trajectory[2,0], c=premier_point_couleur, s=100, edgecolors='w')  # Premier point
ax.scatter(mean_trajectory[0,indice_point_specifique], mean_trajectory[1,indice_point_specifique], mean_trajectory[2,indice_point_specifique], c=point_specifique_couleur, s=100, edgecolors='w')  # Point spécifique
ax.scatter(mean_trajectory[0,-1], mean_trajectory[1,-1], mean_trajectory[2,-1], c=dernier_point_couleur, s=100, edgecolors='w')
# Personnaliser le graphique
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.legend()

plt.savefig("3d_proj.svg",format='svg')
plt.show()
plt.clf()

for i in range(reduction):
    plot_psth(1,2,psth[i][0], psth[i][1], psth[i][2],len(psth[i]))
    plt.savefig(f"psth_dim_{i+1}.svg", format='svg')
    plt.show()
    plt.clf()
