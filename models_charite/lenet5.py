import sys
import scipy
import pathlib
import shutil
import tempfile
import json
import datetime
import numpy as np
from numpy import load
from numpy import save

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from scipy.fft import fftshift  
from scipy.ndimage import convolve1d, convolve 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import meet

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

rand_stat = 42

### Shuffle and split data // .T is required to switch back to shape of (trial x feature)
shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat)
print(shuffled_data.shape)
X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat)
print(X_train.shape)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=rand_stat)
print(X_test.shape)
print(X_val.shape)

##### COMMENT  if performing the while script!
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
#####

## Train and save the model
max_epochs = 500
earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_binary_crossentropy",  
            mode ="min", patience = 10,  
            restore_best_weights = True)
earlystopping_max = tf.keras.callbacks.EarlyStopping(monitor ="val_binary_crossentropy",  
            mode ="max", patience = 10,  
            restore_best_weights = True)

model = keras.Sequential()

model.add(layers.Conv2D(filters=60, kernel_size=(3, 3), input_shape=(240,320,4)))
model.add(layers.LeakyReLU(alpha=0.2))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=120, kernel_size=(3, 3)))
model.add(layers.LeakyReLU(alpha=0.2))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=1, activation='sigmoid'))

model.compile(
	loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
	optimizer='adam', 
	metrics=[
		tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'), 
		'accuracy', 
		tf.metrics.BinaryAccuracy(threshold=0.0)])

history = model.fit(
    x=X_Train[:5000], 
    y=y_train[:5000], 
    epochs=max_epochs, 
    batch_size=64,
    validation_data=(X_Test[:2000], y_test[:2000]),
    callbacks =[earlystopping, earlystopping_max])

model.save('/media/christoph/Volume/Masterthesis/models_trained_and_saved_test_place/le_net_5')
np.save('/media/christoph/Volume/Masterthesis/histories_of_models_trained_and_saved_test_place/le_net_5', history.history)