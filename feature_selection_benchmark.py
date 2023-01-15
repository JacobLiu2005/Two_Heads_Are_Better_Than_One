"""Benchmark with feature selection

This pipeline takes a feature vector and labels, and then process them through feature selection
After feature selection, the new feature vector and labels will be fed into EEGEyeNet Becnhmark to test the performance of ML models
All the results will be store at the location specified in config_FS.py
"""

import sys
sys.path.append('.')
import numpy as np
import logging
import time
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from benchmark import benchmark
from config import config
from hyperparameters_FS import all_models
import os

from plot import plot


def try_models(trainX, trainY, models, save_trail=''):

    all_runs = []
    statistics = []

    
    for name, model in models.items():
        logging.info("Feature selection through " + name)

        model_runs = []


        # create the selection model with the corresponding parameters
        selector = model[0](**model[1])
        logging.info(selector)
        
        all_statistics = []
        # feature selection
        for n_features in selector.n_features:
            if n_features != 258:
                start_time = time.time()
                trainX_new = selector.fit_and_transform(trainX, trainY[:,1], n_features = n_features)

                trainX_new = np.reshape(trainX_new,(trainX_new.shape[0],-1,trainX_new.shape[1]))
                logging.info("Selecting " + str(trainX_new.shape[2]) + " features")
                runtime = (time.time() - start_time)
                logging.info("--- Runtime: %s for seconds ---" % runtime)
            else:
                trainX_new = np.reshape(trainX,(trainX.shape[0],-1,trainX.shape[1]))
                runtime = 0;

            selector_statistics = []

            # run EEGEyeNet Benchmark
            benchmark_runs, benchmark_statistics = benchmark(trainX_new,trainY)

            for k in range(len(benchmark_statistics)):
                selector_statistics.append([name, n_features, runtime])
            
            statistics = np.append(selector_statistics, benchmark_statistics, axis=1)
            total_runtime = [np.array(list(map(float,statistics[:,2]))) + np.array(list(map(float,statistics[:,6])))]

            statistics = np.append(statistics,total_runtime,axis=1)
            if len(all_statistics) == 0:
                all_statistics = statistics
            else:
                all_statistics = np.append(all_statistics, statistics, axis=0)
            
        print(all_statistics.shape)

        # store results and plots
        np.savetxt(config['model_dir']+'/runs_'+name+'.csv', benchmark_runs, fmt='%s', delimiter=',', header='Model,Score,Runtime', comments='')
        np.savetxt(config['model_dir']+'/statistics_'+name+'.csv', all_statistics, fmt='%s', delimiter=',', header='Selection_model,n_features,Selection_runtime,Model,Mean_score,Std_score,Mean_runtime,Std_runtime,Total_runtime', comments='')
        plot(data = all_statistics, fig_name=name, model_name=all_statistics[0,3])

def feature_selection_benchmark(trainX, trainY):
    models = all_models[config['task']][config['dataset']][config['preprocessing']]
    try_models(trainX, trainY, models)

