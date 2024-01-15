import os
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
folder_path = 'C:/Users/Vincent/Documents/GitHub/Data_Analysis/test.csv'
folder_path = '/Users/vincent/Desktop/data Michael/Spike_bins/spikes_bin_1.csv'
# C:\Users\Vincent\Documents\GitHub\Data_Analysis\spikes# Replace with your folder path
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
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data_list.append(row)
    
    numpy_array = np.array(data_list)
    return numpy_array

array = csv_to_numpy(folder_path)
view = input("voulez vous voir les donnees ?")
if view == "oui":
    neurone= input("quel est le neuronne voulu ?")
    print(array[1:,int(neurone)])
    print(array[0])
    x = np.linspace(0, len(array)-1, len(array)-1)  
    array = array[1:,int(neurone)]
    plt.plot(x ,array)
    plt.show() 

if view == "non":
    reduction_type = input("quel type de reduction voulez vous ?")

if reduction_type == "pca":
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

if reduction_type == 'UMAP':
    import umap
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
# oui
    
# # Select 3 dimensions from the data_list
# selected_dimensions = [data_list[i] for i in [0, 2, 4]]  # Replace [0, 2, 4] with the indices of the desired dimensions

# # Perform UMAP dimensionality reduction
# reducer = umap.UMAP(n_components=3)
# embedding = reducer.fit_transform(selected_dimensions)

# # Plot the 3D UMAP
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2])
# plt.show()
