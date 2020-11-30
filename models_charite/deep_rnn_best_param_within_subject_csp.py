from __future__ import print_function 
from numpy import load                                                                                                                                              
import numpy as np                                                                                                                                                  
from os import listdir 
import numpy as np 
from numpy import load 
import sys 

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

import scipy
import scipy.io 
import meet

class TemporalSplit(tf.keras.layers.Layer):
    """Split the input tensor into 8 tensors along the spatial dimension."""


    def call(self, inputs):
        # Expect the input to be 3D and mask to be 2D, split the input tensor into 8
        # subtensors along the time axis (axis 1).
        print("INSIDE OF SPLIT")
        print(inputs)
        print(inputs.shape)
        return tf.split(inputs, 12, axis=2)


    def compute_mask(self, inputs, mask=None):
        # Also split the mask into 8 if it presents.
        if mask is None:
            return None
        return tf.split(mask, 12, axis=2)


rand_stat = 42


def normalize_z(to_be_normalized): 
    return (to_be_normalized - np.mean(to_be_normalized)) / (np.std(to_be_normalized))


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


def calculate_hil_features(transformed):
	hil_dat = scipy.signal.hilbert(transformed, axis=0)
	real_hil_dat = np.real(hil_dat)
	imag_hil_dat = np.imag(hil_dat)
	abs_hil_dat = np.abs(hil_dat)
	angle_hil_dat = np.angle(hil_dat)
	return np.asarray((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat))


def metrics_for_conf_mat(tn, fp, fn, tp):
    """
    Computes for given confusion-matrix entries the metrics
    Sensitivity, Specificity, Accuracy, F1-Score and MCC 
    More info: https://en.wikipedia.org/wiki/Confusion_matrix
    """
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = (2 * tp) / ((2 * tp) + fp + fn)
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (sensitivity, specificity, accuracy, f1_score, mcc)


def build_model_after_deeprnn(in_shape=(250, 8), grad_clip=110, imsize = 32, n_colors = 3, n_timewin = 8, optimizer='adam', learning_rate=0.001, clip_vals=False):
    
    lr = 0.001

    input_layer = tf.keras.Input(shape=in_shape)
#-#-#    split_layer_out = TemporalSplit()(input_layer)
#-#-#
#-#-#    convnets = []
#-#-#    # Build parallel CNNs with shared weights
#-#-#    for i in range(n_timewin):
#-#-#        conv_0 = tf.keras.layers.Conv1D(512, 3, padding='same')(split_layer_out[i])
    conv_0 = tf.keras.layers.Conv1D(512, 3, padding='same')(input_layer)
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
#-#-#        convnets.append(flat)
#-#-#
#-#-#    # Now concatenate the parallel CNNs to one model
#-#-#    concatted = tf.keras.layers.concatenate(convnets)

    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    num_features = 3968
#-#-#    reshaped = tf.keras.layers.Reshape((n_timewin, num_features))(concatted)
    reshaped = tf.keras.layers.Reshape((n_timewin, num_features))(flat)
    lstm = tf.keras.layers.LSTM(128)(reshaped)
    den_0 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(lstm)
    drop_0 = tf.keras.layers.Dropout(0.5)(den_0)
    den_1 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(drop_0)

    model = tf.keras.Model(inputs=input_layer, outputs=den_1)
    print(model.summary())
    
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


for title in ['epoched_intrplt_kx_data_combined',
    'epoched_intrplt_filt_under_100_kx',
    'epoched_intrplt_filt_over_100_kx',
    'epoched_intrplt_filt_over_400_kx',
    'epoched_intrplt_filt_500_900_kx']:
    for k in range(1, 11): 

        model = build_model_after_deeprnn(optimizer='sgd', learning_rate=0.001, clip_vals=True, n_timewin=1, in_shape=(250, 1))

        noise_k00x = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/%s_noise.npy' % (k, title))
        hfsep_k00x = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/%s_hfsep.npy' % (k, title))

        csp_filters, csp_eigenvals = meet.spatfilt.CSP(hfsep_k00x[:8,:,:].mean(2), noise_k00x[:8,:,:].mean(2))
        hfSEP_CSP_0 = np.tensordot(csp_filters[0].T, hfsep_k00x[:8,:,:], axes=(0 ,0))
        noise_CSP_0 = np.tensordot(csp_filters[0].T, noise_k00x[:8,:,:], axes=(0 ,0))

        znorm_noise = normalize_z(noise_CSP_0)
        znorm_hfsep = normalize_z(hfSEP_CSP_0)

        print(znorm_hfsep.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            np.swapaxes(np.concatenate((znorm_hfsep, znorm_noise), axis=1), 1, 0),
            np.concatenate((np.ones(znorm_hfsep.shape[-1], dtype='int8'), np.zeros(znorm_noise.shape[-1], dtype='int8')), axis=0),
            test_size=0.33,
            random_state=rand_stat)

        print(X_train.shape)

        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.33, random_state=rand_stat)

        # Remove dimensions to assure dividability:
        X_train = X_train[:(32 * (X_train.shape[0] // 32))]
        X_test = X_test[:(32 * (X_test.shape[0] // 32))]
        X_eval = X_eval[:(32 * (X_eval.shape[0] // 32))]

        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        X_eval = np.expand_dims(X_eval, axis=2)

        y_train = y_train[:(32 * (y_train.shape[0] // 32))]
        y_test = y_test[:(32 * (y_test.shape[0] // 32))]
        y_eval = y_eval[:(32 * (y_eval.shape[0] // 32))]

        # Our vectorized labels
        y_train = y_train.reshape((-1,1))
        y_test = y_test.reshape((-1,1))
        y_eval = y_eval.reshape((-1,1))

        history = model.fit(x=X_train, y=y_train, epochs=25, batch_size=32, validation_data=[X_eval, y_eval]) 

        model.save('/media/christoph/Volume/Masterthesis/new_model_trained_and_saved/csp/%s/deep_rnn_w_%s_lr_%f_clip_%s_k00%d' % (title, 'sgd', 0.001, 'True', k))
        np.save('/media/christoph/Volume/Masterthesis/histories_new_model_trained_and_saved/csp/%s/deep_rnn_w_%s_lr_%f_clip_%s_k00%d' % (title, 'sgd', 0.001, 'True', k), history.history)

        predictions = np.abs(np.rint(model.predict(x=X_test)))
        confusion_matrix_for_model = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = confusion_matrix_for_model.ravel()
        np.save('/media/christoph/Volume/Masterthesis/histories_of_models_trained_and_saved_test_place_confusion_matrices/csp/%s/deep_rnn_w_sgd_lr_0001_clip_True_k00%d_tn_%d_fp_%d_fn_%d_tp_%d_confusion_matrix' % (title, k, tn, fp, fn, tp), confusion_matrix_for_model)

        metrics = metrics_for_conf_mat(tn, fp, fn, tp)
        np.save('/media/christoph/Volume/Masterthesis/histories_of_models_trained_and_saved_test_place_confusion_matrices/csp/%s/deep_rnn_w_sgd_lr_0001_clip_True_k00%d_tn_%d_fp_%d_fn_%d_tp_%d_metrics' % (title, k, tn, fp, fn, tp), metrics)