FingerMovements:

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from scipy.io import arff 
import pandas as pd
import numpy as np


# Load data
train_dat = arff.loadarff('FingerMovements_TRAIN.arff')
train_dat_pd = pd.DataFrame(train_dat[0])
train_dat_np = np.asarray(train_dat_pd) 

y_train = [0 if x.decode('utf-8') in "b'left'" else 1 for x in train_dat_np[:,1]]
X_train = np.asarray([np.asarray([np.asarray([np.asarray(float(y)) for y in x]) for x in train_dat_np[z,0]]) for z in range(train_dat_np.shape[0])])

test_dat = arff.loadarff('FingerMovements_TEST.arff')
test_dat_pd = pd.DataFrame(test_dat[0])
test_dat_np = np.asarray(test_dat_pd) 

y_test = [0 if x.decode('utf-8') in "b'left'" else 1 for x in test_dat_np[:,1]]
X_test = np.asarray([np.asarray([np.asarray([np.asarray(float(y)) for y in x]) for x in test_dat_np[z,0]]) for z in range(test_dat_np.shape[0])])

# Split data to train, eval, test

rand_stat = 42
shuffled_data, shuffled_labels = shuffle(X_train, y_train, random_state=rand_stat)
X_train, X_eval, y_train, y_eval = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat)