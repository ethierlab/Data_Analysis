import os
import csv
import numpy as np

folder_path = 'C:/Users/Vincent/Documents/GitHub/Data_Analysis/spikes/' # Replace with the actual folder path

first_columns = []
print(os.listdir(folder_path))
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            first_column = [row[0] for row in csv_reader]
            first_columns.append(first_column)

# print(first_columns)
final_csv_path_1 = 'C:/Users/Vincent/Documents/GitHub/Data_Analysis/test_1.csv'
with open(final_csv_path_1, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(first_columns)
final_csv_path = 'C:/Users/Vincent/Documents/GitHub/Data_Analysis/test.csv'  # Replace with the desired final CSV path

with open(final_csv_path, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(first_columns)

final_array = np.array(first_columns).T.tolist()

# print(final_array)
