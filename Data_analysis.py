import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
import h5py
import csv
import os
import pandas as pd
from ipywidgets import interact, IntSlider, Button, VBox
from IPython.display import display

def generate_PSTH_data(sample, t_inf, t_supp):
    """
    Generates Peristimulus Time Histogram (PSTH) data for the given sample within specified time bounds.

    Parameters:
    - sample: The input sample data for which the PSTH is to be calculated.
    - t_inf (float): The lower time bound for generating the PSTH data.
    - t_supp (float): The upper time bound for generating the PSTH data.

    Returns:
    A dictionary containing the following keys and corresponding values:
    - 'Time (s)': A numpy array representing the time intervals from -t_inf to t_supp.
    - 'Lower std': A numpy array representing the lower standard deviation values, calculated using the PSTH function.
    - 'Average': A numpy array representing the average values, calculated using the PSTH function.
    - 'Higher std': A numpy array representing the higher standard deviation values, calculated using the PSTH function.
    """
    column1 = np.linspace(-t_inf, t_supp, len(PSTH(sample)[0]))
    column2 = PSTH(sample)[0]
    column3 = PSTH(sample)[1]
    column4 = PSTH(sample)[2]

    file_data = {
        'Time (s)': column1,
        'Lower std': column2,
        'Average': column3,
        'Higher std': column4
    }
    return file_data

def create_Processed_sig_CSV(input_csv_file, output_csv_file, data_to_append):
    """
    Reads the first 7 rows from an input CSV file, appends specified data, and writes the result to an output CSV file.

    Parameters:
    - input_csv_file (str): The path to the input CSV file from which the first 7 rows will be read.
    - output_csv_file (str): The path to the output CSV file where the appended data will be written.
    - data_to_append (dict): A dictionary containing the new data to append. The keys should be 'Time (s)',
      'Lower std', 'Average', and 'Higher std', and the values should be lists of numerical data.

    The function performs the following steps:
    1. Reads the first 7 rows from the input CSV file.
    2. Appends the new data (specified by data_to_append) to the beginning of the rows.
    3. Writes the appended data, along with the headers ['Time (s)', 'events', 'df_f_signal'],
       to the output CSV file.

    If the input file is not found, an error message is printed. Any other exceptions that occur are also
    caught and printed.
    """
    try:
        # Read the first 6 rows from the input CSV file
        with open(input_csv_file, 'r') as csv_file:
            reader = csv.reader(csv_file)
            rows_to_append = [next(reader) for _ in range(7)]

        # Append the new data at the beginning of the rows to write
        data_to_write = [data_to_append['Time (s)'], data_to_append['events'],
                         data_to_append['df_f_signal']]
        data_to_write = list(zip(*data_to_write))  # Transpose the data
        headers = ['Time (s)', 'events', 'df_f_signal']
        # Write the appended data with headers to the output CSV file
        with open(output_csv_file, 'w', newline='') as csv_output_file:
            writer = csv.writer(csv_output_file)
            writer.writerows(rows_to_append)
            writer.writerow(headers)  # Write the headers
            writer.writerows(data_to_write)

        print(f"Successfully read the first 7 rows from '{input_csv_file}', "
              f"appended new data, and wrote them to '{output_csv_file}'.")
    except FileNotFoundError:
        print(f"Error: File '{input_csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_PSTH_CSV(input_csv_file, output_csv_file, data_to_append):
    """
    Reads the first 7 rows from an input CSV file, appends new data, and writes to an output CSV file.

    Parameters:
    - input_csv_file (str): The path to the input CSV file from which the first 7 rows will be read.
    - output_csv_file (str): The path to the output CSV file where the appended data will be written.
    - data_to_append (dict): A dictionary containing the new data to append. The keys should be 'Time (s)',
      'Lower std', 'Average', and 'Higher std', and the values should be lists of numerical data.

    The function performs the following steps:
    1. Reads the first 7 rows from the input CSV file.
    2. Appends the new data (specified by data_to_append) to the beginning of the rows.
    3. Writes the appended data, along with the headers ['Time (s)', 'Lower std', 'Average', 'Higher std'],
       to the output CSV file.

    If the input file is not found, an error message is printed. Any other exceptions that occur are also
    caught and printed.
    """
    try:
        # Read the first 6 rows from the input CSV file
        with open(input_csv_file, 'r') as csv_file:
            reader = csv.reader(csv_file)
            rows_to_append = [next(reader) for _ in range(7)]

        # Append the new data at the beginning of the rows to write
        data_to_write = [data_to_append['Time (s)'], data_to_append['Lower std'],
                         data_to_append['Average'], data_to_append['Higher std']]
        data_to_write = list(zip(*data_to_write))  # Transpose the data
        headers = ['Time (s)', 'Lower std', 'Average', 'Higher std']
        # Write the appended data with headers to the output CSV file
        with open(output_csv_file, 'w', newline='') as csv_output_file:
            writer = csv.writer(csv_output_file)
            writer.writerows(rows_to_append)
            writer.writerow(headers)  # Write the headers
            writer.writerows(data_to_write)

        print(f"Successfully read the first 7 rows from '{input_csv_file}', "
              f"appended new data, and wrote them to '{output_csv_file}'.")
    except FileNotFoundError:
        print(f"Error: File '{input_csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def csv_to_np_arrays(file_path):
    """
    Reads a CSV file and converts each column into a separate NumPy array.

    Parameters:
    - file_path (str): The path to the CSV file to be read.

    The function reads the CSV file, skipping the first 7 rows, and iterates through each column,
    converting it into a NumPy array. The resulting arrays are stored in a dictionary where the
    keys are the column names, and the values are the corresponding NumPy arrays.

    Note: The function assumes that the first 7 rows of the CSV file do not contain relevant data
    and skips them. Make sure that this assumption aligns with the structure of your CSV file.

    Returns:
    - arrays_dict (dict): A dictionary where keys are the column names, and values are the corresponding
      NumPy arrays.
    """
    # Read the CSV file with pandas, skipping the first 7 rows
    df = pd.read_csv(file_path, skiprows=7)

    # Initialize an empty dictionary to store arrays
    arrays_dict = {}

    # Iterate over each column
    for column in df.columns:
        # Convert column to numpy array and store in dictionary
        arrays_dict[column] = df[column].to_numpy()

    return arrays_dict



def add_lines_to_txt(input_filename, lines, output_filename):
    """
    Adds specified lines to the beginning of a text file, and writes the result to a new file.

    Parameters:
    - input_filename (str): The path to the existing text file whose content will be copied.
    - lines (list of str): A list of strings representing the lines to be added to the beginning of the new file.
    - output_filename (str): The path to the output text file where the final content will be written.

    This function performs the following steps:
    1. Opens a new text file specified by output_filename.
    2. Writes the given lines to the new file.
    3. Copies the content from the existing file (input_filename) to the new file.
    4. Removes the original file (input_filename).

    The result is a new file that contains the specified lines followed by the content of the original file.

    Example:
        lines = ["Header1", "Header2"]
        add_lines_to_txt('input.txt', lines, 'output.txt')
    """
    # Step 1: Open the new text file
    with open(output_filename, 'w') as final:
        # Step 2: Write the given lines to the new file
        for line in lines:
            final.write(line + "\n")

        # Step 3: Write data from the existing CSV file to the new file
        with open(input_filename, 'r') as temp:
            for line in temp:
                final.write(line)
    os.remove(input_filename)

def list_of_lists_to_csv(data, filename):
    """
    Writes a list of lists to a CSV file with specified headers.

    Parameters:
    - data (list of lists): A list of lists containing numerical data, where each inner list represents a column.
      The structure is expected to align with the provided header ["Time", "events", "Isobestic", "GrabDA"].
    - filename (str): The path to the output CSV file where the data will be written.

    This function first transposes the data, as the input is structured with each inner list representing a column
    rather than a row. It then writes the data to the specified CSV file with the header ["Time", "events", "Isobestic", "GrabDA"].
    """
    # Transpose the list of lists using zip
    header = ["Time","events","Isobestic","GrabDA"]
    transposed_data = list(map(list, zip(*data)))

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(transposed_data)

def write_to_csv(filename, data,  channel_titles):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(headers_info)
        writer.writerow(channel_titles)
        writer.writerows(data)



def split_txt_to_csv(filename):
    """
    Splits a given text file into separate CSV files, based on specific experiment data and header information.

    Parameters:
    - filename (str): The path to the input text file containing experiment data.

    The function reads the text file line by line and recognizes specific header indicators to identify different
    experiments. The experiment data is saved into individual CSV files, and the function also handles comments 
    within the data rows.

    The input text file is expected to contain specific header indicators and a tab-delimited structure. Header 
    indicators include "Interval=", "ExcelDateTime=", "TimeFormat=", "DateFormat=", "ChannelTitle=", and "Range=".
    Data rows are expected to be tab-delimited, and comments within data rows must be preceded by a '#'.

    Returns:
    - experiment_count (int): The total number of experiments found and written to CSV files.
    - headers_info (list): A list of header information for the last experiment.
    """
    header_indices = ["Interval=", "ExcelDateTime=", "TimeFormat=", "DateFormat=", "ChannelTitle=", "Range="]
    experiment_count = 1
    data = []
    headers_info = []
    channel_titles = ['Time']
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if any(index in line for index in header_indices):
                if data:  # this means we've collected data for previous experiment and it's time to save it
                    write_to_csv(f'Experiment_{experiment_count}.csv', data, channel_titles)
                    data = []  # reset data
                    headers_info = []  # reset headers_info
                    experiment_count += 1
                if "ChannelTitle=" in line:
                    channel_titles = ['Time'] + line.split("\t")[1:]  # get channel titles from this line
                else:
                    headers_info.append(line)
            else:  # means it's data row
                if '#' in line:  # there's a comment in this line
                    data_part, comment = line.split('#', 1)
                    headers_info.append('#' + comment.strip())
                    line = data_part.strip()
                row = [float(x) for x in line.split('\t')]
                data.append(row)
                
        if data:  # write the final experiment
            write_to_csv(f'Experiment_{experiment_count}.csv', data, channel_titles)
        
        return experiment_count, headers_info



def search_file(PATH:str, channel_events, AnalogIn=False, AnalogOut=False):
    """
    Look for what you need in the doric file

    Arg:
    PATH (str): Path of the .doric file you want to analyze
    channel_events (int): channel where are stock the good events (1, 2, 3 or 4)
    AnalogIn (bool): Put True if you want the AnalogIn files
    AnalogOut (bool): Put True if you want the AnalogOut files

    Return: List of all the files you need
    """
    photometry_doc = h5py.File(PATH, 'r')

    data = photometry_doc['DataAcquisition']['FPConsole']['Signals']['Series0001']

    for i in list(data):
        print(f'{i}: ' + str(list(data[i])))

    res = []

    res.append(np.array(data['AIN01xAOUT01-LockIn']['Time'])) # Time [0]
    res.append(np.array(data['DigitalIO'][f'DIO0{channel_events}'])) # events [1] 
    res.append(np.array(data['AIN01xAOUT01-LockIn']['Values'])) # Isobestic [2]
    res.append(np.array(data['AIN01xAOUT02-LockIn']['Values'])) # GrabdDA [3]

    # # Not commonly used file 
    
    # if AnalogIn:
    #     res.append(np.array(data['AnalogIn']['AIN01'])) # [4]

    # if AnalogOut:
    #     num = input("Which channel of AnalogOut? (1, 2, both): ")
    #     if num == '1' or num == '2':
    #         res.append(np.array(data['AnalogOut'][f'AOUT0{num}'])) # [4] or [5]

    #     elif num == 'both':
    #         res.append(np.array(data['AnalogOut']['AOUT01'])) # [4] or [5]
    #         res.append(np.array(data['AnalogOut']['AOUT02'])) # [5] or [6]

    #     else:
    #         raise ValueError("There's no channel with this name")

    return res


def classify_events(data, sampling_rate, time_window):
    # Change the signal for uniformed 1
    for i,elem in enumerate(data):
        if elem > 0.5:
            data[i] = 1
        if elem < 0:
            data[i] = 0
    
    diff = np.diff(data)
    indices = np.where(diff >= 0.9)[0] + 1
    window_size = int(sampling_rate * time_window)  # Convert time window to number of samples

    # Lists to store the indices of different classes
    single_events = []
    double_events = []
    triple_events = []
    unhandled_events = []
    i = 0
    while i < len(indices):
        window_end = indices[i] + window_size
        count = np.sum((indices >= indices[i]) & (indices < window_end))

        if count == 1:
            single_events.append(indices[i])
        elif count == 2:
            double_events.append(indices[i])
        elif count == 3:
            triple_events.append(indices[i])
        elif count >= 3:
            unhandled_events.append(indices[i]) # utilité pour continuous signal
        i += count  # skip the counted indices
    return single_events, double_events, triple_events, unhandled_events


def find_frequency(raw_time):
    '''
    Find the sampling frequency
    Parameters:
    raw_time (list): list of time 

    Return:
    An integer that represent the sampling frequency
    '''
    return int(1/(raw_time[1] - raw_time[0]))


def denoising(signal, cutoff_freq, kernelsize, sampling_freq):
    """
    Denoise a signal to a certain cutoff frequency and a kernel size to adjust

    Parameters:
    signal (list): The input data
    cutoff_freq (int): cutoff frequency of the low-pass filter
    kernelsize (int): Size of the kernel for the median filter
    sampling_freq (int): Sampling frequency of the signal. You can have this information with the find_frequency() function

    Return: Denoised signal
    """
    # Median filter
    signal_denoised = medfilt(signal, kernel_size=kernelsize)

    # Butterworth filter
    b,a = butter(2, cutoff_freq, btype='low', fs = sampling_freq)
    signal_denoised = filtfilt(b,a, signal_denoised)
    return signal_denoised


def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is, 
                 the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    x = x[np.logical_not(np.isnan(x))]
    X=np.matrix(x)
    m=X.size
    i=np.arange(0,m)
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)


def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,
                 the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    x = x[np.logical_not(np.isnan(x))]
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z

def normalize(signal):
    """
    Normalize the input signal with the z score method

    Parameters:
    signal (list): List of signal values

    Return: Normalized signal with the z score method
    """

    # Find the mean of the signal
    mean = np.nanmean(signal)

    # Find the standard deviation of the signal
    stdev = np.nanstd(signal)

    # Normalize with the z score
    z_score = np.array([((i - mean)/stdev) for i in signal])
    
    return z_score

def motion_correction(grabDA, isobestic, correction):
    """
    Do the motion correction of the signal

    Parameters:
    grabDA (list): List of the grabDA signal values
    isobestic (list): List of the isobestic signal values
    correction (list): List of signal values that you want to correct. Use the function twice to do motion correction on the grabDA and isobestic signal

    Return: Motion corrected signal
    """

    # Build a mask for the NaN
    mask = ~np.isnan(grabDA) & ~np.isnan(isobestic)

    # Make a linear regression with the scipy.stats module
    slope, intercept, r_value, p_value, std_err = linregress(x=grabDA[mask], y=isobestic[mask])

    # Print the slope and r-squared
    print('Slope    : {:.3f}'.format(slope))
    print('R-squared: {:.3f}'.format(r_value**2))

    # Build the motion
    motion = intercept + (slope * isobestic)

    # Substract from the grabDA
    signal_corrected = correction - motion

    return signal_corrected

def artifact_removal(signal, raw_time): ### We must consider the NaN in our analysis
    """
    Remove the artifact from the signal

    Parameters:
    signal (list): Signal to remove artifact from
    raw_time (list): Time list

    Return: Signal without the artifacts 
    """

    # Fit the signal to a polyline
    parameter = np.polyfit(raw_time, signal, deg=4)
    fitted_signal = np.polyval(parameter, raw_time)

    # Find the standard deviation of the signal
    stdev = np.std(signal)

    # Identification of the artifact with 2 * standard deviation to have a 95% confidence interval
    signal_artifact_low, signal_artifact_high = np.where(signal < (fitted_signal - 2*stdev))[0], np.where(signal > (fitted_signal + 2*stdev))[0]
    signal_artifact = np.concatenate((signal_artifact_high, signal_artifact_low))

    # Replace the corresponding index by NaN
    signal[signal_artifact] = np.nan


def find_event_index(signal:list,freq, type=0, sampling_time_window=0,select=0):
    """
    Find the index of peaks and adds them to a list

    Parameters:
    signal (list): list of Signal that contains the peaks to find
    type(int): determine which testing is in use, 0 is for continuous
    sampling_time (int): based as 0, value of time for each set
    freq (int): frequence of data sampling

    Return: List of index where peaks were found
    """
    peaks = []
    if type == 0:
        #single spike signal
        prime = []
        for i in range(1, len(signal)):
            prime.append(signal[i] - signal[i-1])

        peaks = find_peaks(prime)[0]
        return peaks.tolist()
    if type == 1:
        # big amount of spike in determined amount of time
        for i in range(1, len(signal)-1):
        # Check if the current element is a peak
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            # If this is the first peak or the peak is outside the interval from the last peak
                if not peaks or (i - peaks[-1][1] >= sampling_time_window):
                # Add the peak to the list (value, index)
                    peaks.append((signal[i], i))
    # Return only the first peak index of each interval
        return [peak[0] for peak in peaks]

    if type == 2:
        # fonction pour knob turning
        if select == 0:
            return np.concatenate((classify_events(signal, freq, sampling_time_window)[1], classify_events(signal, freq, sampling_time_window)[2]))
        if select == 1:
            return classify_events(signal, freq, sampling_time_window)[1]
        if select == 2:
            return classify_events(signal, freq, sampling_time_window)[2]
        else:
            return classify_events(signal, freq, sampling_time_window)[0]
    if type == 3:
        # CONTINUOUS STIMULATION
        return classify_events(signal, freq, sampling_time_window)[3]
    if type == 4:
        # LEVER TRIAL
        return

def check_time_value(*signal): # Pour les cut de signaux pour faire le psth
    """
    Cut given signals if lists are of differents legnth so they can be of the same dimensions
    
    Parameters:
    
    *signal (list): List representing the signals to analyze
    
    Return: *Signal
    """
    min_length = min([len(i) for i in signal])
    return [i[:min_length] for i in signal]


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
    inf_offset =  int(t_inf * frequency)
    supp_offset = int(t_supp * frequency)

    for val in peak_index:
        if val - inf_offset < 0 and val == min(peak_index):
            raise ValueError("Warning! The time before the events is too large. Change the t_inf parameter.")
        
        if val + supp_offset + 1 > len(brain_stim):
            raise ValueError("Warning! The time after the events is too large. Change the t_supp parameter.")

        stim_cut.append(brain_stim[val - inf_offset : val + supp_offset + 1])

    return stim_cut


def PSTH(stim_cut:list):
    """ 
    Find the average and give standard deviation of each value given 
    
    Parameters:
    Stim_cut (list): list of list of values
    
    Return: list type of values containing 3 arguments, list of average minus standard deviations, list of average and list of average plus standard deviation.
    """
    # Using zip to iterate over the corresponding elements of the sublists
    stats = [(np.mean(values), np.std(values)) for values in zip(*stim_cut)]
    avg_minus_std = [mean - std for mean, std in stats]
    avg = [mean for mean, _ in stats]
    avg_plus_std = [mean + std for mean, std in stats]
    return avg_minus_std, avg, avg_plus_std


def navigate_and_save_subplots(raw_time, event_index, saving_path, folder_name, t_inf, t_supp, frequency, samples):
    """
    Navigates and saves multiple subplots based on the provided parameters.

    Parameters:
    - raw_time (array-like): The time array corresponding to the samples.
    - event_index (array-like): Indices of the events to be plotted.
    - saving_path (str): Path to the directory where the plots will be saved.
    - folder_name (str): Name of the folder where plots will be saved within the directory specified in saving_path.
    - t_inf (float): The lower time bound (in seconds) for the window around each event to plot.
    - t_supp (float): The upper time bound (in seconds) for the window around each event to plot.
    - frequency (float): Sampling frequency of the data.
    - samples (array-like): Data samples to be plotted, corresponding to the events in event_index.

    The function displays an interactive plot with a navigation slider that allows the user to scroll through 
    pages of subplots, each showing a sample plot from the provided samples. A button below each plot page 
    allows the user to save the current page to the specified folder.
    """
    # Define number of plots and plots per page
    num_plots = len(samples)
    plots_per_page = 4

    # Define the maximum page number
    max_page = num_plots // plots_per_page
    if num_plots % plots_per_page != 0:
        max_page += 1  # add one page if there are remaining plots

    def display_subplots(page):
        page = int(page)  # Ensure page is an integer
        plt.close('all')  # close all existing plots
        fig, axs = plt.subplots(1, plots_per_page, figsize=(15, 4), sharey=True)
        for i in range(plots_per_page):
            index = i + (page * plots_per_page)
            if index >= num_plots:
                break
            event_idx = event_index[index]
            start_idx = max(0, int(event_idx - t_inf * frequency))
            end_idx = min(len(raw_time), int(event_idx + t_supp * frequency))
            axs[i].plot(raw_time[start_idx:end_idx + 1], samples[index], color='black')
            axs[i].set_title(f"Sample {index + 1}")
            axs[i].set_xlabel("Time [s]")
            axs[i].set_ylabel("Signal intensity")
        plt.tight_layout()
        plt.show()
    
        # Define what to do when button is clicked
        def on_button_clicked(b):
            # Save the current figure
            script_dir = os.path.dirname(saving_path)
            results_dir = os.path.join(script_dir, folder_name + '/')
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            fig.savefig(results_dir + f"plot_page_{page}")
            plt.close(fig)
        
        # Create save button
        button = Button(description="Save current page")
        button.on_click(on_button_clicked)
        
        # Display button below the plot
        display(button)

    # Create widget to navigate pages
    interact(display_subplots, page=IntSlider(min=0, max=max_page-1, step=1, value=0, continuous_update=False))

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
    plt.show()