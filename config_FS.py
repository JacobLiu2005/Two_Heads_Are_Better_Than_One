import sys
sys.path.append('.')

##########################################################################
##########################################################################
############### FEATURE SELECTION CONFIGURATIONS #########################
##########################################################################
##########################################################################
# 'LR_task' (dataset: 'antisaccade'):
# 'Direction_task' (dataset: 'dots' or 'processing_speed'): dots = "Large Grid Dataset" and processing_speed = "Visual Symbol Search"
# 'Position_task' (dataset: 'dots'):

config_FS = dict()

config_FS['full_data'] = False # whether to use entire dataset
config_FS['first_half'] = False # time segmentation, 1H or 2H data

##################################################################
##################################################################
############### PATH CONFIGURATIONS ##############################
##################################################################
##################################################################
# Where experiment results are stored.
config_FS['log_dir'] = './runs'
# Path to training data.
config_FS['data_dir'] =  '/data' if config_FS['full_data'] else '/data_splitted'
# Path of root
config_FS['root_dir'] = '.' if config_FS['full_data'] else './feature_selection'

# use LR_task dataset
config_FS['task'] = 'LR_task'
config_FS['dataset'] = 'antisaccade'
config_FS['preprocessing'] = 'min' # use minimally preprocessed data
config_FS['feature_extraction'] = True # must be set to True for ML_models operating on feature extracted data

# options of feature selection models
config_FS['univariate_selection'] = True 
config_FS['sequential_selection'] = False     
config_FS['importance_based_selection'] = True 
config_FS['include_your_models'] = False 


def build_file_name():
    if config_FS['full_data']:
        EEG_file = ''
    else:
        if config_FS['first_half']:
            EEG_file = 'Full_First_half_'
        else:
            EEG_file = 'Full_Second_half_'
    EEG_file = EEG_file + config_FS['task'] + '_with_' + config_FS['dataset'] + '_' + 'synchronised_' + config_FS['preprocessing']
    EEG_file = EEG_file + ('_hilbert.npz' if config_FS['feature_extraction'] else '.npz')
    return EEG_file
config_FS['EEG_file'] = build_file_name() # or use your own specified file name
