import urllib.request
import glob
import pandas as pd
from numpy import genfromtxt
import numpy as np

defect = 'Crack'
Models = glob.glob(f'F:/### Sanabil Dissertation/Disso_Scripts/Main_Code/models/{defect}/*')

for model in Models:
    modelname = str(model[65:]).replace(" ", "+")
    modelsavename = str(model[65:])
    starturl = 'http://localhost:6006/data/plugin/scalars/scalars?tag=epoch_accuracy&run='
    endurl = '%5Ctrain&format=csv'
    fullurl = starturl + modelname + endurl
    urllib.request.urlretrieve(fullurl, f"F:/### Sanabil Dissertation/Disso_Scripts/Main_Code/model2csv/Tensorboard/{defect}/{modelsavename}.csv")
