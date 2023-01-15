"""
Data splitting module
Run this code; splitted data will be stored in the assigned directory
"""

import sys
from turtle import shape
sys.path.append('.')

import numpy as np
from config import config, config_FS

from utils import IOHelper
import csv

save_directory = './feature_selection' + '/data_splitted'
save_file_name = config['all_EEG_file']

def split(chopping_indices):
    EEG, labels = IOHelper.get_npz_data(config['data_dir'], verbose=True)
    begin = 0
    firstHalfEEG = []
    secondHalfEEG = []
    firstHalfLabels = []
    secondHalfLabels = []
    for index in chopping_indices:
        end = begin + index
        individual_EEG = EEG[begin:end]
        individual_labels = labels[begin:end]
        if len(firstHalfEEG) == 0:
            firstHalfEEG = np.array_split(individual_EEG,2,axis=0)[0]
            secondHalfEEG = np.array_split(individual_EEG,2,axis=0)[1]
            firstHalfLabels = np.array_split(individual_labels,2,axis=0)[0]
            secondHalfLabels = np.array_split(individual_labels,2,axis=0)[1]
        else:
            firstHalfEEG =  np.append(firstHalfEEG, np.array_split(individual_EEG,2,axis=0)[0], axis=0)
            secondHalfEEG = np.append(secondHalfEEG, np.array_split(individual_EEG,2,axis=0)[1], axis=0)
            firstHalfLabels =  np.append(firstHalfLabels, np.array_split(individual_labels,2,axis=0)[0], axis=0)
            secondHalfLabels = np.append(secondHalfLabels, np.array_split(individual_labels,2,axis=0)[1], axis=0)
        begin += index

    print('Saving data...')
    print("Shapes of first half EEG are: ")
    print(firstHalfEEG.shape)
    print("Shapes of first half labels are: ")
    print(firstHalfLabels.shape)
    print("Shapes of second half EEG are: ")
    print(secondHalfEEG.shape)
    print("Shapes of second half labels are: ")
    print(secondHalfLabels.shape)
    np.savez(save_directory + '/Full_First_half_' + save_file_name, EEG=firstHalfEEG, labels=firstHalfLabels)
    np.savez(save_directory + '/Full_Second_half_' + save_file_name, EEG=secondHalfEEG, labels=secondHalfLabels)
        
def readChopping(file):
    f = open(file, 'r+')
    lines = f.read().split(' \n')
    f.close()
    lines = list(map(int,lines))
    return lines

if __name__=='__main__':
    choppingIndices = readChopping('./feature_selection'+'/lengths.txt')
    split(choppingIndices)