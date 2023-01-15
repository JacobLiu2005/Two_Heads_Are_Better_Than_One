## Two Heads (based on EEGEyeNet)

Our Two Heads method integrates ML/DL models with feature selection to evaluate ET prediction based on EEG measurements with varying difficulty levels. 

## Overview

This repository consists of general functionality to run the Two Heads method and customize the implementation of different feature selection models and classifiers. We offer to run standard feature selection pipelines (univariate selection, etc.) and ML models (e.g., kNN, SVR, etc.) to benchmark our method. The feature selection modules can be found in FeatureSelection_Models directory under the feature_selection folder. The classifiers' implementation can be found in the StandardML_Models directory.

The data we use for classification only consists of one task from the EEGEyeNet dataset: 
LR (left-right)

To download the data, click here: https://osf.io/download/jkrzh/

## Installation (Environment)

There are many dependencies in this benchmark, and we propose to use anaconda as a package manager.

To install the entire environment to run all models (standard machine learning and deep learning models in both PyTorch and TensorFlow), use the eegeyenet_benchmark.yml file. To do so, run:

```bash
conda env create -f twoheads_benchmark.yml
```

Otherwise, you can create a minimal environment to run only the models you want to try (see the following section).

### General Requirements

Create a new conda environment:

```bash
conda create -n twoheads_benchmark python=3.8.5 
```

First, install the general_requirements.txt

```bash
conda install --file general_requirements.txt 
```

### Pytorch Requirements

If you want to run the PyTorch DL models, install PyTorch in the recommended way below. For Linux users with GPU support, the command is:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 
```

For other installation types and Cuda versions, visit [pytorch.org](https://pytorch.org/get-started/locally/).

### Tensorflow Requirements

If you want to run the TensorFlow DL models, run.

```bash
conda install --file tensorflow_requirements.txt 
```

### Standard ML Requirements

If you want to run the standard ML models, run:

```bash
conda install --file standard_ml_requirements.txt 
```

These requirements should be installed after PyTorch to avoid dependency issues.

## Configuration of Feature Selection

The feature selector configuration takes place in hyperparameters_FS.py under the feature_selection directory. The Two Heads pipeline configuration is contained in config_FS.py.

### config_FS.py

We start by explaining the settings that can be made for running the pipeline:

Choose the task to run in the benchmark:

```bash
config_FS['task'] = 'LR_task'
```

Choose whether to recognize processed EEGEyeNet data using the Hilbert transformation. Set to True for the standard ML models:

```bash
config_FS['feature_extraction'] = True
```

Choose to run feature selection on the entire dataset:
```bash
config_FS['full_data'] = True
```

Choose to run feature selection on 1H data:
```bash
config_FS['first_half'] = True
```

Choose to run feature selection on 2H data:
```bash
config_FS['first_half'] = False
```

Include our univariate (filter-based) selection models in the pipeline run:

```bash
config_FS['univariate_selection'] = True 
```

Include the embedded methods models into the pipeline run:

```bash
config_FS['importance_based_selection'] = True
```

Include the sequential selection (wrapper-based) models into the pipeline run:

```bash
config_FS['sequential_selection'] = True
```


Include your feature selection models as specified in hyperparameters_FS.py. For instructions on how to create your custom models, see further below.

```bash
config_FS['include_your_models'] = True
```



### hyperparameters_FS.py

Here we define our feature selectors. They are configured in a dictionary containing the object of the selectors and hyperparameters passed when the object is instantiated.

You can add your selectors in the your_selectors dictionary. Make sure to enable all the models you want to run in config_FS.py.



## Configuration of Classification

The classifier configuration takes place in hyperparameters.py. The training configuration is contained in config.py.

### config.py

We start by explaining the settings that can be made for running the benchmark:

Choose the task to run in the benchmark:

```bash
config['task'] = 'LR_task'
```

Choose data preprocessed with Hilbert transformation. Set to True for the standard ML models:

```bash
config['feature_extraction'] = True
```

To include our standard ML models in the benchmark run:

```bash
config['include_ML_models'] = True 
```

To include our deep learning models in the benchmark run:

```bash
config['include_DL_models'] = True
```

Include your models as specified in hyperparameters.py. For instructions on how to create your custom models, see further below.

```bash
config['include_your_models'] = True
```

Include dummy models for comparison against the benchmark:

```bash
config['include_dummy_models'] = True
```

You can either choose to train models or use existing ones in /run/ and perform inference with them. To train custom models, set the following parameters:

```bash
config['retrain'] = True 
config['save_models'] = True 
```

Set both to False if you want to load existing models and perform inference. 
In this case, specify the path to your existing model directory under

```bash
config['load_experiment_dir'] = path/to/your/model 
```

You can run our deep learning models in both PyTorch and TensorFlow. You can specify which framework you want to use in the model configuration section. Just specify it in config.py, make sure you set up the environment as explained above, and everything specific to the framework will be handled in the background.

config.py also allows to configure hyperparameters such as the learning rate and enables early stopping of models.

### hyperparameters.py

Here we define our models. Standard ML and deep learning models are configured in a dictionary containing the object of the model and hyperparameters that are passed when the object is instantiated.

You can add your models in the your_models dictionary. Specify the models for each task separately. Make sure to enable all the models you want to run in config.py.

## Running the benchmark

Create a directory that stores raw/preprocessed data before the split into 1H and 2H, and specify its location in config.py, e.g.

```bash
config['data_dir'] = './data/'
```

Create a ./runs directory to save files while running our Two-Heads benchmark.

### ./feature_selection/split_data.py

To obtain 1H and 2H data, run:

```bash
python3 ./feature_selection/split_data.py
```

### ./feature_selection/feature_selection_benchmark.py

In feature_selection_benchmark.py, we load all selectors specified in hyperparameters_FS.py and one single classifier specified in hyperparameters.py. Each selector outputs selected features, and the classifier is trained and then evaluated with the scoring function corresponding to the benchmarked task.

A current run directory is created, containing a training log, saving console output, and model checkpoints of all runs.

### ./feature_selection/main.py

To start the Two Heads pipeline, run:

```bash
python3 ./feature_selection/main.py
```
A current run directory is created, containing a training log, saving console output, and model checkpoints of all runs.

## Add Custom Models

To benchmark feature selectors, we use a common interface we call selector in feature_selection_benchmark.py. A selector is an object that implements the following methods:

```bash
fit() 
transform()
fit_and_transform()
```

To benchmark classifiers, we use a common interface we call trainer in benchmark.py. A trainer is an object that implements the following methods:

```bash
fit() 
predict() 
save() 
load() 
```

### Implementation of custom models

To implement your custom model, make sure that you create a class that implements the above methods. If you use library models, wrap them into a class that implements the above interface used in our pipeline.

### Adding custom models to the Two Heads pipeline

You can add objects that implement the above interface. Make sure to enable your custom models in config_FS.py and config.py. In hyperparameters_FS.py, add your custom feature selectors into the your_models dictionary. For custom classifiers, add them into the your_models dictionary in hyperparameters.py.

