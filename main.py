import sys
from config_FS import config_FS

sys.path.append('.')

from config import config, create_folder
import numpy as np

import logging
import time
from utils import IOHelper
import csv
from benchmark import benchmark
import IOHelper_FS
from feature_selection_benchmark import feature_selection_benchmark

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2



if __name__ == '__main__':
    create_folder()

    logging.basicConfig(filename=config['info_log'], level=logging.INFO)
    logging.info('Started the Logging')
    logging.info(f"Using {config['framework']}")
    start_time = time.time()

    trainX, trainY = IOHelper_FS.get_npz_data(config_FS['root_dir'] + config_FS['data_dir'], verbose=True)
    feature_selection_benchmark(trainX, trainY)

    logging.info("--- Runtime: %s seconds ---" % (time.time() - start_time))
    logging.info('Finished Logging')
