import os
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
import umap
import data_analysis as da
folder_path = 'C:/Users/Vincent/Downloads/Recording 1'
# folder_path = '/Users/vincent/Desktop/data Michael/Spike_bins/spikes_bin_1.csv'
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
    numpy_arrays = []
    for column in transposed_data:
        try:
            float_column = np.array(column, dtype=float)
            numpy_arrays.append(float_column)

        except ValueError:
            numpy_arrays.append(np.array(column))

    return numpy_arrays

array = csv_to_numpy(folder_path)
action = input("voulez vous voir un seul neuronne ?")

if action == "oui":
    neurone= input("quel est le neuronne voulu ?")
    
    selected_row = np.array(array[int(neurone)-1][1:], dtype=float)

    x = np.arange(len(selected_row))
    plt.plot(x, selected_row)
    plt.show()
    action = input("voulez vous créer un nouveau csv pour ce neurone ?")
    if action == "oui":
        with open(f'neuronne_{neurone}.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow(selected_row)
action = input("voulez vous voir la visualisation 3D ?")
if action == "oui":
    reduction_type = input("quel type de reduction voulez vous (pca/UMAP?")

if reduction_type in ("pca", "PCA", "Pca", "pCa", "pcA", "PcA", "pCA", "PCa"):
    pca = PCA(n_components=3)
    data_transformed = pca.fit_transform(array)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_transformed[:, 0], data_transformed[:, 1], data_transformed[:, 2])

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('Visualisation 3D avec PCA')
    plt.show()
    action = input("Voulez vous sauvegarder les données ?")
    if action == "oui":
        with open('pca.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            spamwriter.writerow(data_transformed)
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

    
    
