import os
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
import umap
import data_analysis as da
import pandas as pd

class StateMachine():
    def __init__(self, path):
        self.states = {
            'read_data': self.state_read_data,
            'process_data': self.state_process_data,
            'single_neurone': self.state_single_neurone,
            'psth': self.state_psth,
            'pca': self.state_pca,
            # Other states can be added here
        }
        self.array = self.state_read_data(path)
        self.current_state = None

    def set_state(self, name):
        self.current_state = name

    def run(self):
        try:
            state_function = self.states[self.current_state]
        except KeyError:
            raise ValueError("Invalid state: " + self.current_state)

        # Run the current state function and update the state
        new_state = state_function()
        self.set_state(new_state)
    
    def read_csv_and_transform(self, file_path, base_change_matrix):
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
    def csv_to_numpy(self, file_path):
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
    def plot_pca_variance(self, variance_explained):
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
    
    def read_csv_and_transform(self,file_path, base_change_matrix):
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
    
    def process_folder_for_psth(self, folder_path, base_change_matrix, column_index):
            all_column_data = []
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    transformed_data = self.read_csv_and_transform(file_path, base_change_matrix)
                    if transformed_data is not None:
                        all_column_data.append(transformed_data[:, column_index])

            # Trouver la longueur de la liste la plus courte
            min_length = min(map(len, all_column_data))

            # Tronquer toutes les listes à la longueur de la liste la plus courte
            truncated_data = [col[:min_length] for col in all_column_data]

            return truncated_data
    def state_read_data(self):
        pass
    def state_pca(self):
        pass
        
        
    def state_psth(self):
        pass
        
    
    def state_single_neurone():
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
        return 'analyze_data'  # Transition to the next state





def state_analyze_data():
    # Code to analyze data
    # ...
    return 'visualize_data'  # Transition to the next state

def state_visualize_data():
    # Code to visualize data
    # ...
    return 'end'  # End of the state machine process

def state_end():
    return 'end'  # Keeps the state machine at the end

# Instantiate the state machine and add states
sm = StateMachine()

# Starting state
sm.set_state('read_data')

# Example of how to run the state machine
while sm.current_state != 'end':
    sm.run()