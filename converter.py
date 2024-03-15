import os
import pandas as pd
def convert_txt_to_csv(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return
    # List all the files in the given directory
    files = os.listdir(directory_path)
    for file_name in files:
        if file_name.endswith(".txt"):
            txt_path = os.path.join(directory_path, file_name)
            # Reading the txt file into a dataframe
            df = pd.read_csv(txt_path, delimiter="\t",skiprows=6, header=None)  # Assuming tab-separated values in txt
            # Converting .txt extension to .csv
            csv_path = os.path.join(directory_path, file_name.replace(".txt", ".csv"))
            # Saving the dataframe as a csv file
            df.to_csv(csv_path, index=False)
            print(f"Converted '{file_name}' to .csv format.")
if __name__ == '__main__':
    #changer le directory pour le chemin a convertir les fichier
    # print(os.path.exists("/Users/freddydagenais/Desktop/Maitrise/code/235/electrique/exp1"))
    #oublies pas de rajouter un "/" post path

    directory = "/Users/vincent/Desktop/fred/230/Droit"
    convert_txt_to_csv(directory)