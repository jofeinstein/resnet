import pickle
import numpy as np
import pandas as pd
import os
from os import listdir
import glob

feature = pickle.load(open('/Users/jofeinstein/Desktop/features/train/1/1hqcB00-3/1hqcB00_blended_XOY+_r0_OO.pickle', 'rb'))
path = '/Users/jofeinstein/Desktop/features/train/1/*/*.pickle'
#print(feature.size)
data = pd.DataFrame()
df = pd.DataFrame()

count = 0
for dillpickle in glob.glob(path):
    count += 1
    print(count)
    if count == 1:
        feature_vec = pickle.load(open(dillpickle, 'rb'))
        data = pd.DataFrame(feature_vec)
    elif count == 100:
        break
    else:
        feature_vec = pickle.load(open(dillpickle, 'rb'))
        df = pd.DataFrame(feature_vec)
        data = pd.merge(data, df)

print(df.shape)
print(data.shape)