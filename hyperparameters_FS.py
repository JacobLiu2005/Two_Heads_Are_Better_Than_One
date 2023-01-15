from FeatureSelection_Models.ImportanceBasedSelection import ImportanceBasedSelector
from FeatureSelection_Models.SequentialSelection import SequentialSelector
from FeatureSelection_Models.UnivariateSelector import UnivariateSelector

from hyperparameters import merge_models
from config_FS import config_FS
import numpy as np

# Use the following formats to add models (see hyperparameters_FS.py for examples)
# 'NAME' : [MODEL, {'n_features': [...], 'param1' : value1, 'param2' : value2, ...}]
# the model should be callable with MODEL(param1=value1, param2=value2, ...)

your_selectors = {
    'LR_task' : {
        'antisaccade' : {
            'max' : {
                
            },

            'min' : {
                
            }
        }
    },
}

our_univariate_selection_models = {
    'LR_task' : {
        'antisaccade' : {
            'max' : {
                'f_classif_uni' : [UnivariateSelector, {'scoring_name': 'f_classif'}]
            },

            'min' : {
                'f_classif_uni' : [UnivariateSelector, {'scoring_name': 'f_classif', 'n_features': [131]}],
                # 'f_regression' : [UnivariateSelector, {'scoring_name': 'f_regression', 'n_features': np.append(np.arange(1,258,10),258)}],
                # 'mutual_info_classif' : [UnivariateSelector, {'scoring_name': 'mutual_info_classif', 'n_features': np.append(np.arange(1,258,10),258)}],
                # 'mutual_info_regression' : [UnivariateSelector, {'scoring_name': 'mutual_info_regression', 'n_features': np.append(np.arange(1,258,10),258)}]
            }
        }
    },
}

our_sequential_selection_models = {
    'LR_task' : {
        'antisaccade' : {
            'max' : {
                'LinearSVC_seq' : [SequentialSelector, {'estimator_name': 'LinearSVC', 'C': 0.01, 'penalty': "l1", 'dual': False}]
            },

            'min' : {
                'LinearSVC_seq' : [SequentialSelector, {'estimator_name': 'LinearSVC', 'C': 0.01, 'penalty': "l1", 'dual': False, 'n_features': np.arange(1,258,10)}]
            }
        }
    },
}

our_importance_based_selection_models = {
    'LR_task' : {
        'antisaccade' : {
            'max' : {
                'LinearSVC_imp' : [ImportanceBasedSelector, {'estimator_name': 'LinearSVC', 'C': 0.01, 'penalty': "l1", 'dual': False}]
            },

            'min' : {
                'LinearSVC_imp' : [ImportanceBasedSelector, {'estimator_name': 'LinearSVC', 'C': 0.01, 'penalty': "l1", 'dual': False, 'n_features': np.append(np.arange(1,258,10),258)}],
                'Lasso' : [ImportanceBasedSelector, {'estimator_name': 'Lasso', 'alpha' : 0.1, 'n_features': np.append(np.arange(1,258,10),258)}],
                'LogisticRegression' : [ImportanceBasedSelector, {'estimator_name': 'LogisticRegression', 'random_state': 0, 'n_features':  np.append(np.arange(1,258,10),258)}],
                'RandomForest' : [ImportanceBasedSelector, {'estimator_name': 'RandomForest', 'max_depth': 10, 'n_estimators': 250, 'n_jobs' : -1, 'n_features': np.append(np.arange(1,258,10),258)}],
                'DecisionTree' : [ImportanceBasedSelector, {'estimator_name': 'DecisionTree', 'max_depth': 5, 'random_state': 0, 'n_features': np.append(np.arange(1,258,10),258)}],
                'ExtraTrees' : [ImportanceBasedSelector, {'estimator_name': 'ExtraTrees', 'n_estimators': 50, 'n_features':  np.append(np.arange(1,258,10),258)}],
                'Ridge' : [ImportanceBasedSelector, {'estimator_name': 'Ridge', 'alphas': np.logspace(-6, 6, num=5), 'n_features': np.append(np.arange(1,258,10),258)}]
            }
        }
    },
}


all_models = dict()

if config_FS['univariate_selection']:
    all_models = merge_models(all_models, our_univariate_selection_models)
if config_FS['sequential_selection']:
    all_models = merge_models(all_models, our_sequential_selection_models)
if config_FS['importance_based_selection']:
    all_models = merge_models(all_models, our_importance_based_selection_models)
if config_FS['include_your_models']:
    all_models = merge_models(all_models, your_selectors)