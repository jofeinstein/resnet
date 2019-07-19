import pickle
import numpy as np
import pandas as pd
import os
from os import listdir
import glob

feature = pickle.load(open('/home/jfeinst/Projects/resnet-files/features/train/1/1hqcB00-3/1hqcB00_blended_XOY+_r0_OO.pickle', 'rb'))
path = '/home/jfeinst/Projects/resnet-files/features/train/1/*/*.pickle'
#print(feature.size)
#df = pd.DataFrame(feature)


for pickle in glob.glob(path):
    feature_vec = open(pickle, 'rb')
    print(type(feature_vec))