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


rand_stat = 42


def normalize_z(to_be_normalized): 
    return (to_be_normalized - np.mean(to_be_normalized)) / (np.std(to_be_normalized))


def calculate_hil_features(transformed):
    hil_dat = scipy.signal.hilbert(transformed, axis=0)
    real_hil_dat = np.real(hil_dat)
    imag_hil_dat = np.imag(hil_dat)
    abs_hil_dat = np.abs(hil_dat)
    angle_hil_dat = np.angle(hil_dat)
    return np.concatenate((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat), axis=0)


#-#for title in ['epoched_intrplt_kx_data_combined',
#-#    'epoched_intrplt_filt_under_100_kx',
#-#    'epoched_intrplt_filt_over_100_kx',
#-#    'epoched_intrplt_filt_over_400_kx',
#-#    'epoched_intrplt_filt_500_900_kx']:
title = 'epoched_intrplt_filt_500_900_kx'
for k in range(1, 11):

    model = tf.keras.models.load_model('/media/christoph/Volume/Masterthesis/new_model_trained_and_saved/%s/deep_rnn_w_%s_lr_%f_clip_%s_k00%d' % (title, 'sgd', 0.001, 'True', k))

    noise_k00x = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/%s_noise.npy' % (k, title))
    hfsep_k00x = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/%s_hfsep.npy' % (k, title))

    znorm_noise = normalize_z(noise_k00x[:8, :, :])
    znorm_hfsep = normalize_z(hfsep_k00x[:8, :, :])

    _, X_test, _, y_test = train_test_split(
    np.swapaxes(np.concatenate((znorm_hfsep, znorm_noise), axis=2), 2, 0),
    np.concatenate((np.ones(znorm_hfsep.shape[-1], dtype='int8'), np.zeros(znorm_noise.shape[-1], dtype='int8')), axis=0),
    test_size=0.33,
    random_state=rand_stat)

    X_test = X_test[:(32 * (X_test.shape[0] // 32))]
    y_test = y_test[:(32 * (y_test.shape[0] // 32))]

    # Our vectorized labels
    y_test = y_test.reshape((-1,1))

    predictions = np.abs(np.rint(model.predict(x=X_test)))
    confusion_matrix_for_model = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = confusion_matrix_for_model.ravel()
    print(title + ' for ' + str(k))
    print((tn, fp, fn, tp))
    np.save('/media/christoph/Volume/Masterthesis/new_model_trained_and_saved/confusion_matrices/%s_deep_rnn_w_%s_lr_%f_clip_%s_k00%d_tn_%d_fp_%d_fn_%d_tp_%d_confusion_matrix' % (title, 'sgd', 0.001, 'True', k, tn, fp, fn, tp), confusion_matrix_for_model)