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

# C:/Users/Vincent/Documents/GitHub/Data_Analysis/spikes# Replace with your folder path
# csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# data_list = []
# for file in csv_files:
#     file_path = os.path.join(folder_path, file)
#     data = np.genfromtxt(file_path, delimiter=',', usecols=0)
#     data_list.append(data.tolist())

# print(data_list)

# reshaped_data = np.array(data_list).reshape(-1, 1)
def csv_to_numpy(file_path):
    if not os.path.exists(file_path):
        print(f"Le fichier ou le dossier {file_path} n'existe pas.")
        return None
    data_list = []

    def process_file(file_path):
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader, None)  # Skip the header if it exists
            for row in csv_reader:
                data_list.append(row)

    def process_directory(directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                process_file(item_path)
            elif os.path.isdir(item_path):
                process_directory(item_path)  # Recursive call

    if os.path.isfile(file_path):
        process_file(file_path)
    elif os.path.isdir(file_path):
        process_directory(file_path)
    
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
def plot_psth(psth_results, dimension):
    psth_low, psth_mean, psth_high = psth_results
    indices = range(len(psth_mean))

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
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.savefig(f'psth_{dimension + 1}.png')
    # Afficher le graphique
    plt.show()

def plot_variance_cloud(ax, mean, std_dev_plus, std_dev_minus):
    # Créer un nuage de points pour la variance
    X = np.concatenate([mean, mean])
    Y = np.concatenate([std_dev_plus, std_dev_minus])
    Z = np.concatenate([std_dev_plus, std_dev_minus])
    ax.scatter(X, Y, Z, color='red', alpha=0.1)
def create_variance_plane(ax, base, spread, axis):
    u = np.linspace(np.min(base), np.max(base), 100)
    v = np.linspace(-1, 1, 10)
    U, V = np.meshgrid(u, v)
    if axis == 'x':
        X = base
        Y = spread[0] * V + base
        Z = spread[1] * V + base
    elif axis == 'y':
        X = spread[0] * V + base
        Y = base
        Z = spread[1] * V + base
    elif axis == 'z':
        X = spread[0] * V + base
        Y = spread[1] * V + base
        Z = base
    
    # Ajouter une surface pour visualiser la variance
    ax.plot_surface(X, Y, Z, color='red', alpha=0.1, linewidth=0)
folder_path = 'C:/Users/Vincent/Downloads/Recording 1'
# folder_path = "C:/Users/Vincent/Downloads/Recording 1/spikes_bin_1.csv"
# folder_path = '/Users/vincent/Desktop/data Michael/Spike_bins/spikes_bin_1.csv'

array = csv_to_numpy(folder_path)

action = input("voulez-vous voir un seul neurone ?")

if action == "oui":
    neurone= input("quel est le neurone voulu ?")
    
    selected_row = array[:, int(neurone)]

    x = np.arange(len(selected_row))
    plt.plot(x, selected_row)
    plt.show()
    action = input("voulez-vous créer un nouveau csv pour ce neurone ?")
    if action == "oui":
        with open(f'neuronne_{neurone}.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow(selected_row)
action = input("voulez vous voir le PSTH de chaque neurone ?")
if action == "oui":
    for neurone in range(array.shape[1]):
        all_data = process_csv_files_in_folder(folder_path, neurone)
        psth_results = da.PSTH(all_data)
        
        plot_psth(psth_results,neurone)
action = input("voulez vous voir faire une réduction de dimensionalité ?")
if action == "oui":
    reduction_type = input("quel type de reduction voulez vous (pca/UMAP?")
reduction_type = 'pca'
if reduction_type in ("pca", "PCA", "Pca", "pCa", "pcA", "PcA", "pCA", "PCa"):
    print(array.shape)
    # Calculer le PCA et montre la variance expliquée
    pca = PCA().fit(array)
    variance = pca.explained_variance_ratio_ * 100
    # plot_pca_variance(variance)

    reduction = input("combien de dimension voulez-vous conserver ?")
    reduction = 3
    # Réduire les données à X dimensions
    pca = PCA(n_components=int(reduction))
    data_transformed = pca.fit_transform(array)
    
    
    
    action = input("voulez vous voir le PSTH de chaque dimension ?")
    if action == "oui":
        psth = []
        matrix = pca.components_
        for dimension in range(matrix.T.shape[1]):
            all_data = process_folder_for_psth(folder_path, dimension,matrix.T)
            psth_results = da.PSTH(all_data)
            psth.append(psth_results)
            plot_psth(psth_results,dimension)
    psth = None
    # action = input("Voulez vous sauvegarder les données ?")
    # if action == "oui":
    #     with open('pca.csv', 'w', newline='') as csvfile:
    #         spamwriter = csv.writer(csvfile, delimiter=',')
    #         spamwriter.writerow(data_transformed)

    action = input("voulez vous voir la visualisation 3D ?")
    action = 'oui'
    if action == "oui":
        if psth is None:
            psth = []
            matrix = pca.components_
            for dimension in range(matrix.T.shape[1]):
                all_data = process_folder_for_psth(folder_path, dimension,matrix.T)
                psth_results = da.PSTH(all_data)
                # plot_psth(psth_results,dimension)
                psth.append(psth_results)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(psth[0][1], psth[1][1], psth[2][1], label='Mean Trajectory')
        mean_trajectory = np.array([psth[0][1], psth[1][1], psth[2][1]])  # Une matrice 3 x N des points de la trajectoire moyenne
        std_x = psth[0][2][0] - psth[0][1][0]  # Écart-type pour la dimension x
        std_y = psth[1][2][0] - psth[1][1][0]  # Écart-type pour la dimension y
        std_z = psth[2][2][0] - psth[2][1][0]
        def plot_ellipses(ax, trajectory, std_x, std_y, std_z):
            # Nombre de points pour représenter l'ellipse
            num_points = 100  
            theta = np.linspace(0, 2 * np.pi, num_points)

            # Tracer une ellipse à chaque point de la trajectoire
            for i in range(trajectory.shape[1]):
                # Coordonnées du centre de l'ellipse
                x_center, y_center, z_center = trajectory[:, i]

                # Vecteurs pour l'ellipse
                x_ellipse = std_x * np.cos(theta)
                y_ellipse = std_y * np.sin(theta)

                # Tracer l'ellipse
                ax.plot(x_center + x_ellipse, y_center + y_ellipse, z_center + std_z * np.ones_like(theta), color='r', alpha=0.1)

        # Appeler la fonction pour tracer les ellipses le long de la trajectoire
        plot_ellipses(ax, mean_trajectory, std_x, std_y, std_z)

        
        # plot_variance_cloud(ax, psth[0][1], psth[0][2], psth[0][0])
        # plot_variance_cloud(ax, psth[1][1], psth[1][2], psth[1][0])  
        # plot_variance_cloud(ax, psth[2][1], psth[2][2], psth[2][0])
        
        # create_variance_plane(ax, psth[0][1], (psth[0][0], psth[0][2]), 'x')
        # create_variance_plane(ax, psth[1][1], (psth[1][0], psth[1][2]), 'y')
        # create_variance_plane(ax, psth[2][1], (psth[2][0], psth[2][2]), 'z')

        # Personnaliser le graphique
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
        ax.legend()

        plt.show()
        
if reduction_type in ("umap", "UMAP", "Umap", "uMap", "umAp", "umaP", "uMAP", "UmAP", "UMaP", "UMAp", "uMaP", "uMaP"):
    
    reducer = umap.UMAP(n_components=3)
    embedding = reducer.fit_transform(array)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2])

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('Visualisation 3D avec UMAP')
    plt.show()

    
    
