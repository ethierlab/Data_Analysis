import os
import csv
import numpy as np

# folder_path = 'C:/Users/Vincent/Documents/GitHub/Data_Analysis/spikes/' # Replace with the actual folder path
folder_path = '/Users/vincent/Downloads/spike_bins_csvs'

first_columns = []
# print(os.listdir(folder_path))
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            first_column = [row[0] for row in csv_reader]
            first_columns.append(first_column)

# print(first_columns)
final_csv_path = '/Users/vincent/Desktop/CSV_spikes_305/CSV_spikes_305.csv'
with open(final_csv_path, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(first_columns)

for i in range(len(first_columns)):
    print(len(first_columns[i]))

final_array = np.array(first_columns).T.tolist()

# print(final_array)
