import data_analysis as da
import os
import numpy as np
import matplotlib.pyplot as plt

chan_events = "Stim" # Enter the channel name for the Events
Signal_psth = "EMG flex."  # Enter the channel name for the PSTH
experiment = 0 # Put 1 for single spike signal, 2 for knob exepriement, 3 for continuous stimulation, 4 for lever experiement
t_inf = 0 # (seconds) Enter time value before stimulation
t_supp = 0.001 # (seconds) Enter time value after stimulation
select = 0 # Select for succes & failure (0), success (1), failure (2), stimulation (4)
time_window = 0.3 # (seconds) Enter the sampling time between two spikes of "starting squence"
select = 0 # If trial with success and failure are displayed,0 is for every signal, 1 is for success, 2 for failure, 3 is for activation stimulation
path_donnees= "/Users/freddydagenais/Desktop/Maitrise/code/230/opto_peripherique /230_stim_opto_periph_droit_csv/"
path_saving = "/Users/freddydagenais/Desktop/Maitrise/code/230/opto_peripherique/230_stim_opto_periph_droit_csv/psth/" # Chemin ou tu veux save tes données
def read_text_file_in_current_folder():
    files = os.listdir(path_donnees)
    print(files)

    for file_name in files:
        if file_name.endswith(".csv"):
            PATH = path_donnees + "/" + file_name
            dictio = da.csv_to_np_arrays(PATH)
            keys_list = []
            # Add keys from the dictionary to the list
            for key in dictio.keys():
                keys_list.append(key)
                try:
                    time =  dictio[keys_list[0]]
                    events = dictio[chan_events]
                    y_sig = dictio[Signal_psth]
                except KeyError:
                    print("The channel name you entered you entered does not exist")

            # plt.figure(figsize=(30, 6))
            # plt.plot(time, events)
            # plt.show()

            freq = da.find_frequency(time)
            index = da.find_event_index(events,freq, select, time_window,experiment)
            # print(index)
            # plt.plot(time,events)
            # plt.scatter(time[index],events[index], color="g")
            # plt.show()

            sample = da.cut_individual_event(t_inf, t_supp, index, y_sig, freq)
            # da.navigate_and_save_subplots(time, index,path_saving, "plots", t_inf, t_supp, freq, sample)

            da.plot_psth(t_inf,t_supp,*da.PSTH(sample), len(sample))

            data = da.generate_PSTH_data(sample, t_inf, t_supp)
            filename = PATH[PATH.rfind("/")+1:-4] + "_PSTH.csv"
            da.create_PSTH_CSV(PATH, path_saving+ filename, data)


            
if __name__ == '__main__':
    read_text_file_in_current_folder()