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
import pickle
from scipy.fft import fftshift  
from scipy.ndimage import convolve1d, convolve 

from pyts.image import RecurrencePlot


rand_stat = 42


hfsep_dat = str(sys.argv[1])
noise_dat = str(sys.argv[2])
title = str(sys.argv[3])
idx = int(sys.argv[4])


def normalize_z(to_be_normalized): 
    return (to_be_normalized - np.mean(to_be_normalized)) / (np.std(to_be_normalized))


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


args = sys.argv
print('Number of arguments: %d arguments.' % len(args))
print('Argument List:', str(args))

rand_stat = 42

def eigh(cov1, cov2):
    rank = np.linalg.matrix_rank(cov2)
    w, v = np.linalg.eigh(cov2)
    # get whitening matrix
    W = v[:,-rank:]/np.sqrt(w[-rank:])
    cov1_white = W.T.dot(cov1).dot(W)
    eigvals, eigvect = np.linalg.eigh(cov1_white)
    return (
            np.sort(eigvals)[::-1],
            W.dot(eigvect)[:,np.argsort(eigvals)[::-1]])

def bCSTP(data1, data2, num_iter, t_keep, s_keep):
    n_ch, n_dp, n_trials = data1.shape
    t_keep = np.r_[n_dp,
            np.linspace(t_keep, n_dp, num_iter).astype(int)[::-1]]
    s_keep = np.linspace(s_keep, n_ch, num_iter).astype(int)[::-1]
    T_FILT = [np.eye(n_dp)]
    S_FILT = []
    S_EIGVAL = []
    T_EIGVAL = []
    for i in range(num_iter):
        print('bCSTP-iteration num %d' % (i + 1))
        # obtain spatial filter
        temp1 = np.tensordot(T_FILT[-1][:,:t_keep[i]], data1, axes=(0,1))
        temp2 = np.tensordot(T_FILT[-1][:,:t_keep[i]], data2, axes=(0,1))
        cov1 = np.einsum('ijl, ikl -> jk', temp1, temp1)
        cov2 = np.einsum('ijl, ikl -> jk', temp2, temp2)
        w, v = eigh(cov1, cov2)
        S_FILT.append(v)
        S_EIGVAL.append(w)
        # obtain temporal filter
        temp1 = np.tensordot(S_FILT[-1][:,:s_keep[i]], data1, axes=(0,0))
        temp2 = np.tensordot(S_FILT[-1][:,:s_keep[i]], data2, axes=(0,0))
        cov1 = np.einsum('ijl, ikl -> jk', temp1, temp1)
        cov2 = np.einsum('ijl, ikl -> jk', temp2, temp2)
        w, v = eigh(cov1, cov2)
        T_FILT.append(v)
        T_EIGVAL.append(w)
    return S_EIGVAL, T_EIGVAL, S_FILT, T_FILT[1:]


def calculate_hil_features(transformed):
  hil_dat = scipy.signal.hilbert(transformed, axis=0)
  real_hil_dat = np.real(hil_dat)
  imag_hil_dat = np.imag(hil_dat)
  abs_hil_dat = np.abs(hil_dat)
  angle_hil_dat = np.angle(hil_dat)
  return np.concatenate((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat), axis=0)


def load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_path, noise_path, title, identifier):
  # load data in format of (channel x epoch_length x number of epochs)
  title_of_run = title % (identifier)
  hfSEP = load(hfsep_path % (identifier))
  noise = load(noise_path % (identifier))

  ### 01_ToDo: Modify all the 11 base-modalities through: [SSD Y/N] (load respective modifiers therefore)
  # Still will remain open. Only thing to do here however: Load non-epoched data, compute SSD, epoch data
  raw_title = title_of_run + '_raw'

  ### 02_ToDo: Modify all the 22 modalities through: [leave, CSP, CCAr, bCSTP]
  # Compute CSP
  csp_title = title_of_run + '_CSP'
  csp_filters, csp_eigenvals = meet.spatfilt.CSP(hfSEP[:8,:,:].mean(2), noise[:8,:,:].mean(2))
  hfSEP_CSP_0 = np.tensordot(csp_filters[0].T, hfSEP[:8,:,:], axes=(0 ,0))
  noise_CSP_0 = np.tensordot(csp_filters[0].T, noise[:8,:,:], axes=(0 ,0))
  hfSEP_CSP_1 = np.tensordot(csp_filters[1].T, hfSEP[:8,:,:], axes=(0 ,0))
  noise_CSP_1 = np.tensordot(csp_filters[1].T, noise[:8,:,:], axes=(0 ,0))
  hfSEP_CSP_2 = np.tensordot(csp_filters[2].T, hfSEP[:8,:,:], axes=(0 ,0))
  noise_CSP_2 = np.tensordot(csp_filters[2].T, noise[:8,:,:], axes=(0 ,0))

  # Compute CCAr
  ccar_title = title_of_run + '_CCAr'
  a_ccar, b_ccar, s_ccar = meet.spatfilt.CCAvReg(hfSEP[:8,:,:])
  ccar_filt_hfSEP_0 = np.tensordot(a_ccar[:,0], hfSEP[:8,:,:], axes=(0, 0))
  ccar_filt_noise_0 = np.tensordot(a_ccar[:,0], noise[:8,:,:], axes=(0, 0))
  ccar_filt_hfSEP_1 = np.tensordot(a_ccar[:,1], hfSEP[:8,:,:], axes=(0, 0))
  ccar_filt_noise_1 = np.tensordot(a_ccar[:,1], noise[:8,:,:], axes=(0, 0))
  ccar_filt_hfSEP_2 = np.tensordot(a_ccar[:,2], hfSEP[:8,:,:], axes=(0, 0))
  ccar_filt_noise_2 = np.tensordot(a_ccar[:,2], noise[:8,:,:], axes=(0, 0))

  # Compute bCSTP
  # s_bcstp_eigenvals, t_bcstp_eigenvals, W_bcstp, V_bcstp = bCSTP(hfSEP[:8,:,:], noise[:8,:,:], num_iter=15, t_keep=3, s_keep=3)
  # left out as it would also need intrplt.data.... the scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_kx_data_combined_hfsep[-1][:,0], intrplt_kx_data_combined[:8]), V_out_epoched_intrplt_kx_data_combined_hfsep[-1][:,0][::-1], axis=-1)
  
  ### 03_ToDo: Modify all the 88 modalities through: [hil Y/N]
  hil_csp_title = title_of_run + '_CSP_hil'
  hil_extracted_hfSEP_CSP_0 = calculate_hil_features(hfSEP_CSP_0)
  hil_extracted_noise_CSP_0 = calculate_hil_features(noise_CSP_0)
  hil_extracted_hfSEP_CSP_1 = calculate_hil_features(hfSEP_CSP_1)
  hil_extracted_noise_CSP_1 = calculate_hil_features(noise_CSP_1)
  hil_extracted_hfSEP_CSP_2 = calculate_hil_features(hfSEP_CSP_2)
  hil_extracted_noise_CSP_2 = calculate_hil_features(noise_CSP_2)
  hil_extracted_CSP_hfSEP = np.concatenate((hil_extracted_hfSEP_CSP_0, hil_extracted_hfSEP_CSP_1, hil_extracted_hfSEP_CSP_2), axis=0)
  hil_extracted_CSP_noise = np.concatenate((hil_extracted_noise_CSP_0, hil_extracted_noise_CSP_1, hil_extracted_noise_CSP_2), axis=0)

  hil_ccar_title = title_of_run + '_CCAR_hil'
  hil_extracted_ccar_filt_hfSEP_0 = calculate_hil_features(ccar_filt_hfSEP_0)
  hil_extracted_ccar_filt_noise_0 = calculate_hil_features(ccar_filt_noise_0)
  hil_extracted_ccar_filt_hfSEP_1 = calculate_hil_features(ccar_filt_hfSEP_1)
  hil_extracted_ccar_filt_noise_1 = calculate_hil_features(ccar_filt_noise_1)
  hil_extracted_ccar_filt_hfSEP_2 = calculate_hil_features(ccar_filt_hfSEP_2)
  hil_extracted_ccar_filt_noise_2 = calculate_hil_features(ccar_filt_noise_2)
  hil_extracted_ccar_hfSEP = np.concatenate((hil_extracted_ccar_filt_hfSEP_0, hil_extracted_ccar_filt_hfSEP_1, hil_extracted_ccar_filt_hfSEP_2), axis=0)
  hil_extracted_ccar_noise = np.concatenate((hil_extracted_ccar_filt_noise_0, hil_extracted_ccar_filt_noise_1, hil_extracted_ccar_filt_noise_2), axis=0)

  hfsep_labels = np.ones(len(hfSEP[0,0,:]), dtype=np.int8)
  noise_labels = np.zeros(len(noise[0,0,:]), dtype=np.int8)

  # return the datasets in epoch, channel, time_in_channel - fashion
  return [
    [np.concatenate((hfSEP[5]-hfSEP[0], noise[5]-noise[0]), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), raw_title],
#   [np.concatenate((hfSEP.reshape(-1, hfSEP.shape[-1]), noise.reshape(-1, noise.shape[-1])), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), title_of_run + '_all_channels_flattened'],// I have no idea why, but this singular error still occurs
    [np.concatenate((hfSEP_CSP_0, noise_CSP_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), csp_title],
    [np.concatenate((ccar_filt_hfSEP_0, ccar_filt_noise_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), ccar_title],
    [np.concatenate((hil_extracted_CSP_hfSEP, hil_extracted_CSP_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_csp_title],
    [np.concatenate((hil_extracted_ccar_hfSEP, hil_extracted_ccar_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_ccar_title]
  ]


def create_alex_net(input_len):
  model = model = keras.models.Sequential([ 
    keras.layers.Conv1D(filters=96, kernel_size=(11), strides=(4), activation='relu', input_shape=(input_len,1)), 
    keras.layers.BatchNormalization(), 
    keras.layers.MaxPool1D(pool_size=(3), strides=(2)), 
    keras.layers.Conv1D(filters=256, kernel_size=(5), strides=(1), activation='relu', padding="same"), 
    keras.layers.BatchNormalization(), 
    keras.layers.MaxPool1D(pool_size=(3), strides=(2)), 
    keras.layers.Conv1D(filters=384, kernel_size=(3), strides=(1), activation='relu', padding="same"), 
    keras.layers.BatchNormalization(), 
    keras.layers.Conv1D(filters=384, kernel_size=(1), strides=(1), activation='relu', padding="same"), 
    keras.layers.BatchNormalization(), 
    keras.layers.Conv1D(filters=256, kernel_size=(1), strides=(1), activation='relu', padding="same"), 
    keras.layers.BatchNormalization(), 
    keras.layers.MaxPool1D(pool_size=(3), strides=(2)), 
    keras.layers.Flatten(), 
    keras.layers.Dense(4096, activation='relu'), 
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(4096, activation='relu'), 
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(1024, activation='relu'), 
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(1, activation='sigmoid') 
  ])

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

  model.compile(optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9, clipvalue=5.0),  
    loss=tf.keras.losses.BinaryCrossentropy(), 
    metrics=METRICS)

  return model


#-#-#workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_dat, noise_dat, title, idx)
#-#-#
#-#-#for data, labels, run_title in workload:
#-#-#  print('run_title: %s' % run_title)
#-#-#  print(data.shape)
#-#-#
#-#-#  ### Shuffle and split data // .T is required to switch back to shape of (trial x feature)
#-#-#  shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat)
#-#-#  print(shuffled_data.shape)
#-#-#  X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat)
#-#-#  print(X_train.shape)
#-#-#  X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.33, random_state=rand_stat)
#-#-#
#-#-#  X_train = np.expand_dims(X_train, axis=-1)
#-#-#  X_test = np.expand_dims(X_test, axis=-1)
#-#-#  X_eval = np.expand_dims(X_eval, axis=-1)
#-#-#
#-#-#  y_train = y_train.reshape((-1,1))
#-#-#  y_test = y_test.reshape((-1,1))
#-#-#  y_eval = y_eval.reshape((-1,1))
#-#-#
#-#-#  # some model definition and training
#-#-#  model = create_alex_net(X_train.shape[1])
#-#-#
#-#-#  history = model.fit(x=X_train, y=y_train, epochs=25, batch_size=32, validation_data=[X_eval, y_eval]) 
#-#-#
#-#-#  model.save('/media/christoph/Volume/Masterthesis/alex_net/models/%d/alex_net_on_%s' % (idx, run_title.replace('/', '-')))
#-#-#  np.save('/media/christoph/Volume/Masterthesis/alex_net/history/%d/alex_net_on_%s' % (idx, run_title.replace('/', '-')), history.history)
#-#-#
#-#-#  predictions = np.abs(np.rint(model.predict(x=X_test)))
#-#-#  confusion_matrix_for_model = confusion_matrix(y_test, predictions)
#-#-#  tn, fp, fn, tp = confusion_matrix_for_model.ravel()
#-#-#  np.save('/media/christoph/Volume/Masterthesis/alex_net/%d/alex_net_on_%s_confusion_matrix' % (idx, run_title.replace('/', '-')), confusion_matrix_for_model)
#-#-#
#-#-#  metrics = metrics_for_conf_mat(tn, fp, fn, tp)
#-#-#  np.save('/media/christoph/Volume/Masterthesis/alex_net/%d/alex_net_on_%s_metrics' % (idx, run_title.replace('/', '-')), metrics)


ssd_modalities = [
  ['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_under_100_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_under_100_kx_noise.npy', 'k00%d_ssd_filt_intrplt_filt_under_100'] , 
  ['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_100_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_100_kx_noise.npy', 'k00%d_ssd_filt_intrplt_filt_over_100'] , 
  ['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_400_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_400_kx_noise.npy', 'k00%d_ssd_filt_intrplt_filt_over_400'] , 
  ['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_500_900_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_500_900_kx_noise.npy', 'k00%d_ssd_filt_intrplt_filt_500_900']
]

for hfsep_dat, noise_dat, title in ssd_modalities:

  # load data in format of (channel x epoch_length x number of epochs)
  identifier = idx
  title_of_run = title % (identifier + 1)
  hfSEP = load(hfsep_dat % (identifier + 1))
  noise = load(noise_dat % (identifier + 1))

  ### 01_ToDo: Modify all the 11 base-modalities through: [SSD Y/N] (load respective modifiers therefore)
  # Still will remain open. Only thing to do here however: Load non-epoched data, compute SSD, epoch data
  raw_title = title_of_run + '_raw'

  hil_ssd_title = title_of_run + '_hil'
  hil_extracted_ssd_filt_hfSEP_0 = calculate_hil_features(hfSEP)
  hil_extracted_ssd_filt_noise_0 = calculate_hil_features(noise)

  hfsep_labels = np.ones(len(hfSEP[0,:]), dtype=np.int8)
  noise_labels = np.zeros(len(noise[0,:]), dtype=np.int8)

  workload = [
    [np.concatenate((hfSEP, noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), raw_title],
    [np.concatenate((hil_extracted_ssd_filt_hfSEP_0, hil_extracted_ssd_filt_noise_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_ssd_title]
  ]

  for data, labels, run_title in workload:
    print('run_title: %s' % run_title)
    print(data.shape)

    ### Shuffle and split data // .T is required to switch back to shape of (trial x feature)
    shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat)
    print(shuffled_data.shape)
    X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat)
    print(X_train.shape)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.33, random_state=rand_stat)

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    X_eval = np.expand_dims(X_eval, axis=-1)

    y_train = y_train.reshape((-1,1))
    y_test = y_test.reshape((-1,1))
    y_eval = y_eval.reshape((-1,1))

    # some model definition and training
    model = create_alex_net(X_train.shape[1])
    history = model.fit(x=X_train, y=y_train, epochs=25, batch_size=32, validation_data=[X_eval, y_eval]) 
    model.save('/media/christoph/Volume/Masterthesis/alex_net/models/%d/alex_net_on_%s' % (idx, run_title.replace('/', '-')))
    np.save('/media/christoph/Volume/Masterthesis/alex_net/history/%d/alex_net_on_%s' % (idx, run_title.replace('/', '-')), history.history)
    predictions = np.abs(np.rint(model.predict(x=X_test)))
    confusion_matrix_for_model = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = confusion_matrix_for_model.ravel()
    np.save('/media/christoph/Volume/Masterthesis/alex_net/%d/alex_net_on_%s_confusion_matrix' % (idx, run_title.replace('/', '-')), confusion_matrix_for_model)
    metrics = metrics_for_conf_mat(tn, fp, fn, tp)
    np.save('/media/christoph/Volume/Masterthesis/alex_net/%d/alex_net_on_%s_metrics' % (idx, run_title.replace('/', '-')), metrics)