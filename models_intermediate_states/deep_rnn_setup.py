from __future__ import print_function 
from numpy import load                                                                                                                                              
import numpy as np                                                                                                                                                  
from os import listdir 
import numpy as np 
from numpy import load 
import sys 

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#-#-# X_Test = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_test.npy', allow_pickle=True)[:,29:-26,40:-82,:] / 255.0 
#-#-# y_test = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_test_labels.npy', allow_pickle=True) 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, models 
 
import tensorflow_docs as tfdocs 
import tensorflow_docs.modeling 
import tensorflow_docs.plots 
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

import time 
 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
 
import numpy as np 
np.random.seed(1234) 
from functools import reduce 
import math as m 
 
import scipy.io 

class TemporalSplit(tf.keras.layers.Layer):
    """Split the input tensor into 8 tensors along the spatial dimension."""

    def call(self, inputs):
        # Expect the input to be 3D and mask to be 2D, split the input tensor into 8
        # subtensors along the time axis (axis 1).
        print("INSIDE OF SPLIT")
        print(inputs)
        print(inputs.shape)
        return tf.split(inputs, 8, axis=2)

    def compute_mask(self, inputs, mask=None):
        # Also split the mask into 8 if it presents.
        if mask is None:
            return None
        return tf.split(mask, 8, axis=2)

rand_stat = 42
noise_k003 = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k003/epoched_intrplt_filt_500_900_kx_noise.npy')
hfsep_k003 = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k003/epoched_intrplt_filt_500_900_kx_hfsep.npy')

def normalize_z(to_be_normalized): 
    return (to_be_normalized - np.mean(to_be_normalized)) / (np.std(to_be_normalized))

znorm_noise = normalize_z(noise_k003[:8, :, :])
znorm_hfsep = normalize_z(hfsep_k003[:8, :, :])

X_train, X_test, y_train, y_test = train_test_split(
   np.swapaxes(np.concatenate((znorm_hfsep, znorm_noise), axis=2), 2, 0),
   np.concatenate((np.ones(znorm_hfsep.shape[-1], dtype='int8'), np.zeros(znorm_noise.shape[-1], dtype='int8')), axis=0),
   test_size=0.33,
   random_state=rand_stat)

X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.33, random_state=rand_stat)

# swap axes for chan x data:
#-#X_train = np.swapaxes(X_train, 1, 2)
#-#X_test = np.swapaxes(X_test, 1, 2)
#-#X_eval = np.swapaxes(X_eval, 1, 2)

# Remove dimensions to assure dividability:
X_train = X_train[:(32 * (X_train.shape[0] // 32))]
X_test = X_test[:(32 * (X_test.shape[0] // 32))]
X_eval = X_eval[:(32 * (X_eval.shape[0] // 32))]

y_train = y_train[:(32 * (y_train.shape[0] // 32))]
y_test = y_test[:(32 * (y_test.shape[0] // 32))]
y_eval = y_eval[:(32 * (y_eval.shape[0] // 32))]

# Our vectorized labels
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))
y_eval = y_eval.reshape((-1,1))

def local_reformatInput(data, labels, trainIndices, validIndices, testIndices):
    """
    Receives the the indices for train and test datasets.
    Outputs the train, validation, and test data and label datasets.
    """

    # Shuffling training data
    shuffledIndices = np.random.permutation(len(trainIndices))
    trainIndices = trainIndices[shuffledIndices]

    print('data ndim: %d' % data.ndim)

    if data.ndim == 4:
        return [(data[trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[validIndices], np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[testIndices], np.squeeze(labels[testIndices]).astype(np.int32))]
    elif data.ndim == 5:
        return [(data[:, trainIndices], np.squeeze(labels[trainIndices]).astype(np.int32)),
                (data[:, validIndices], np.squeeze(labels[validIndices]).astype(np.int32)),
                (data[:, testIndices], np.squeeze(labels[testIndices]).astype(np.int32))]

def build_model_after_deeprnn(in_shape=(250, 8), grad_clip=110, imsize = 32, n_colors = 3, n_timewin = 8, optimizer='adam', learning_rate=0.001, clip_vals=False):
    
    lr = 0.001

    input_layer = tf.keras.Input(shape=in_shape)
    split_layer_out = TemporalSplit()(input_layer)

    convnets = []
    # Build parallel CNNs with shared weights
    for i in range(n_timewin):
        conv_0 = tf.keras.layers.Conv1D(512, 3, padding='same')(split_layer_out[i])
        conv_1 = tf.keras.layers.Conv1D(512, 3, padding='same')(conv_0)
        conv_2 = tf.keras.layers.Conv1D(512, 3, padding='same')(conv_1)
        conv_3 = tf.keras.layers.Conv1D(512, 3, padding='same')(conv_2)
        max_0 = tf.keras.layers.MaxPool1D()(conv_3)
        conv_4 = tf.keras.layers.Conv1D(256, 3, padding='same')(max_0)
        conv_5 = tf.keras.layers.Conv1D(256, 3, padding='same')(conv_4)
        max_1 = tf.keras.layers.MaxPool1D()(conv_5)
        conv_6 = tf.keras.layers.Conv1D(128, 3, padding='same')(max_1)
        max_2 = tf.keras.layers.MaxPool1D()(conv_6)
        flat = tf.keras.layers.Flatten()(max_2)
        convnets.append(flat)

    # Now concatenate the parallel CNNs to one model
    concatted = tf.keras.layers.concatenate(convnets)

    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    num_features = 3968
    reshaped = tf.keras.layers.Reshape((n_timewin, num_features))(concatted)
    lstm = tf.keras.layers.LSTM(128)(reshaped)
    den_0 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(lstm)
    drop_0 = tf.keras.layers.Dropout(0.5)(den_0)
    den_1 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(drop_0)

    model = tf.keras.Model(inputs=input_layer, outputs=den_1)
    print(model.summary())

#-#-#    OLD: concatente the 1D-channels to a MC-Model
#-#-#    combined_model = tf.keras.layers.concatenate(
#-#-#        [tf.keras.backend.expand_dims(channl.output, axis=1) for channl in channel_models]
#-#-#        , axis=1
#-#-#    )
    
    METRICS = [
       tf.keras.metrics.TruePositives(),
       tf.keras.metrics.FalsePositives(),
       tf.keras.metrics.TrueNegatives(),
       tf.keras.metrics.FalseNegatives(), 
       tf.keras.metrics.BinaryAccuracy(),
       tf.keras.metrics.Precision(),
       tf.keras.metrics.Recall(),
       tf.keras.metrics.AUC(),
       tf.keras.metrics.MeanAbsoluteError(),
    ]

    if optimizer is 'adam':
      print("Compiling with ADAM")
      if clip_vals:
        print("Doing it with clipvals")
        model.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0), 
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=METRICS)
      else:
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), 
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=METRICS)
    else:
      print("Compiling with SGD")
      if clip_vals:
        print("Doing it with clipvals")
        model.compile(optimizer=tf.keras.optimizers.SGD(lr, momentum=0.9, clipvalue=5.0), 
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS)
      else:
        model.compile(optimizer=tf.keras.optimizers.SGD(lr, momentum=0.9), 
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS)

    return model

#-#-#(X_train_split, y_train_split), (X_val_split, y_val_split), (X_test_split, y_test_split) = local_reformatInput(X_Test[:3000], y_test[:3000], np.arange(1000), np.arange(1000), np.arange(1000)) 
#-#-#X_train_split = X_train_split.astype("float32", casting = 'unsafe') 
#-#-#X_val_split = X_val_split.astype("float32", casting = 'unsafe') 
#-#-#X_test_split = X_test_split.astype("float32", casting = 'unsafe')

for exp in [['adam', 0.001, False], ['adam', 0.05, False], ['adam', 0.5, False], ['adam', 1.0, False], ['adam', 0.001, True], ['adam', 0.05, True], ['adam', 0.5, True], ['adam', 1.0, True], 
  ['sgd', 0.001, False], ['sgd', 0.05, False], ['sgd', 0.5, False], ['sgd', 1.0, False], ['sgd', 0.001, True], ['sgd', 0.05, True], ['sgd', 0.5, True], ['sgd', 1.0, True]]:

  model = build_model_after_deeprnn(optimizer=exp[0], learning_rate=exp[1], clip_vals=exp[2])

  history = model.fit(x=X_train, y=y_train, epochs=25, batch_size=32, validation_data=[X_eval, y_eval]) 

  model.save('/media/christoph/Volume/Masterthesis/models_trained_and_saved_test_place/deep_rnn_w_%s_lr_%f_clip_%s' % (exp[0], exp[1], str(exp[2])))
  np.save('/media/christoph/Volume/Masterthesis/histories_of_models_trained_and_saved_test_place/deep_rnn_w_%s_lr_%f_clip_%s' % (exp[0], exp[1], str(exp[2])), history.history)