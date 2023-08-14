# Data Analysis

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contribution](#contribution)
- [Notebooks](#notebooks)


## Description

Welcome to the first release of the Multi-Notebook Data Analysis Project! This is a comprehensive solution designed to facilitate data analysis workflows in an efficient and user-friendly manner. The project is a collection of distinct Jupyter Notebooks, each dedicated to specific facets of data analysis.

The project offers a variety of functionalities ranging from data converting to preprocessing and cleansing. Leveraging the power of Python libraries like numpy and matplotlib, it provides analysts with a robust platform for exploratory data analysis and visualization.

Each notebook in the project serves a unique purpose. They are modular and can be used individually based on the specific requirements of your data analysis pipeline. Whether you need to clean your data, perform exploratory data analysis or just convert simple data set into CSV, these notebooks have got you covered.

Being in its first release, this project is far from its complete stage. Our aim is to make data analysis as seamless as possible, thereby enabling you to focus on deriving insights from your data. Please feel free to contribute to the project, suggest improvements, or report issues. Your participation is what will help this project grow and improve!

## Installation

Before running the project, there are a few dependencies that must be installed. The `packages_install.py` script will automatically install all necessary Python packages. To run this script, navigate to the project's directory in your terminal and type the following command:

```bash
python install_packages.py
```
Or it is possible to just double click on the file and the installation will be done automatically.

## Usage
Once the dependencies are installed, you can start using the diffrents jupyter notebooks program. The structure of the code is made so the each different notebooks has a different usage. Each are described in the following section. you can execute each cell depending on what you intend to do with your data. All the function that are called in the differents notebooks are in the `Data_analysis.py` file. 
<!-- Here, describe how the program is used. Include all steps necessary to use it, and any code samples if necessary. -->


## Notebooks

### Data_converting.ipynb
This specific notebook is used to convert data from LabChart exported as .txt and from doric files into a CSV standard file with lines that contains information of the specific experiment.
### Data_Processing.ipynb
In the data_processing notebook, we do the processing of the signals for PSTH usage. We first apply a low-pass filter and a median filter to have the lower frequency in the signal. The user can change the parameter of the low-pass filter as he wants. Then, we do the detrending of the signals with the airPLS algorithm that is an adaptive iteratively reweighted Penalized Least Squares (airPLS) to do the baseline fitting. We only subtract the baseline from the signals to have detrended signals. After that, we normalize the signals with the z-score method, and we correct them by the motion correction method. The motion correction is a comparison of the data from the grabDA and isosbestic signals. We can then fit a line in the data and subtract the line equation from the signals to remove the fluctuation from the isosbestic to the grabDA signal. At the end of the processing, we do the dF/F signal by subtracting the isosbestic from the grabDA to have the response of the neuron to an external light stimulus. At the end of the notebook, we save the signal in a csv file.
### PSTH_Graph.ipynb
This Jupyter notebook is used to visualize and assert the specific channels you want to use to draw a Post Stimulus Time Histogram (PSTH). Also, this files allow you to save the data of the PSTH and also to have the different graphs of stimulations saved as a group of 4.