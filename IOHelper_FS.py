import numpy as np
from config_FS import config_FS
import logging


def get_npz_data(data_dir, verbose = True):
    """Function for reading .npz files
    Return EEG signals and labels as two arrays
    """
    if verbose:
        logging.info('Loading ' + config_FS['EEG_file'])
    with np.load(data_dir + '/' + config_FS['EEG_file']) as f:
        EEG = f['EEG']
        EEG = EEG.reshape((EEG.shape[0],EEG.shape[2]))
        
        if verbose:
            logging.info("X training loaded.")
            logging.info(EEG.shape)
        labels = f['labels']
        if verbose:
            logging.info("y training loaded.")
            logging.info(labels.shape)
    return EEG,labels