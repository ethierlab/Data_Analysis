import data_analysis as da
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_csv_files(folder_path):
    # Initialize an empty list to store the matrix
    matrix = []

    # Iterate through each file in the folder
    for file in os.listdir(folder_path):
        # Check if the file is a CSV
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            row = df.columns.tolist()
            matrix.append(row)
    # Truncate rows to the shortest row length
    min_row_length = min(len(row) for row in matrix)
    print(f"Minimum number of rows across files: {min_row_length}")
    # Truncate each column (list) to the length of the shortest column in the matrix
    min_col_length = min(len(col) for row in matrix for col in row)
    print(f"Minimum number of colums across files: {min_col_length}")
    for i in range(len(matrix)):
        matrix[i] = [col[:min_col_length] for col in matrix[i][:min_row_length]]

    matrix = list(map(list, zip(*matrix)))
    
    
    return matrix

folder_path = 'C:/Users/Vincent/Downloads/Recording 1'
matrix = read_csv_files(folder_path)
PSTH_res = [da.PSTH(column) for column in matrix]

for i, result in enumerate(PSTH_res):
    avg_minus_std, avg, avg_plus_std = result
    da.plot_psth(avg_minus_std, avg, avg_plus_std)
