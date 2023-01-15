
import sys
import time
import logging
import pandas as pd
import numpy as np
import seaborn as sns
from utils import IOHelper
from config import config, create_folder

colors = {0:'orange', 1:'blue'}
trainX, trainY = IOHelper.get_npz_data(config['data_dir'], verbose=True)
trainY = trainY[:,1]
trainX = np.reshape(trainX,(-1,258))
pdX = pd.DataFrame(trainX)
pdX = pdX.rename(columns=lambda x: "Feature ranked "+ str(x+1))
pdY = pd.Series(trainY)
pdY = pdY.rename("Label")
df = pd.concat([pdX,pdY],axis=1)

sns.set_theme(style="ticks")

swarm_plot = sns.histplot(df, x="Feature ranked 1", hue="Label", element="poly")

fig = swarm_plot.get_figure()
fig.savefig("out_second_half_hist.png") 