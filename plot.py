from logging import root
from typing import overload
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from config_FS import config_FS
import time

from config import config

# plot the results for a single half data, 1H or 2H
def plot(data, fig_name, model_name):
    n_features = list(map(int, data[:,1]))
    scores = list(map(float, data[:,4]))

    fig = plt.figure(dpi=110)
    ax = fig.add_subplot( 111 )
    ax.plot(n_features,scores, 'b')
    plt.ylim([min(scores)-0.05,1])
    plt.ylabel("Mean score")
    plt.xlabel("Number of features")
    plt.savefig(config['model_dir']+'/plot_'+fig_name+'_'+model_name+'.png')
    # plt.close()

def readcsv(path):
    data = pd.read_csv(path)
    return data

# plot for two halves, 1H and 2H
def plot_multi(first_half, second_half):
    fig_time = str(int(time.time()))

    fig = plt.figure(dpi=110)
    ax = fig.add_subplot( 111 )
    ax.plot(first_half['n_features'].tolist(),first_half['Mean_score'].tolist(), 'b', label = "first_half_data")
    ax.plot(second_half['n_features'].tolist(),second_half['Mean_score'].tolist(), 'r', label = "second_half_data")

    plt.ylim([min(min(first_half['Mean_score'].tolist()), min(second_half['Mean_score'].tolist()))-0.05, 1])
    plt.legend(loc='upper right')
    plt.ylabel("Mean score")
    plt.xlabel("Number of features")
    plt.savefig(config_FS['root_dir'] + '/plots/' + fig_time + '_' + first_half['Selection_model'][0] + '_selection_' + first_half['Model'][0] + '.png')
    plt.show()

if __name__ == '__main__':
    first_half = readcsv('E:\EEGEyeNet-main\Qi_Richard\\runs\\1657613898_LR_task_antisaccade_min_first\statistics_f_classif_uni.csv')
    second_half = readcsv('E:\EEGEyeNet-main\Qi_Richard\\runs\\1657614148_LR_task_antisaccade_min_second\statistics_f_classif_uni.csv')
    plot_multi(first_half, second_half)
    