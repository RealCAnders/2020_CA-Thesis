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
  return np.stack((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat), axis=0)


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
    [np.concatenate((hfSEP[:8], noise[:8]), axis=-1), np.concatenate((hfsep_labels, noise_labels), axis=0), raw_title],
#   [np.concatenate((hfSEP.reshape(-1, hfSEP.shape[-1]), noise.reshape(-1, noise.shape[-1])), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), title_of_run + '_all_channels_flattened'],// I have no idea why, but this singular error still occurs
    [np.concatenate((np.stack((hfSEP_CSP_0, hfSEP_CSP_1, hfSEP_CSP_2), axis=0), np.stack((noise_CSP_0, noise_CSP_1, noise_CSP_2), axis=0)), axis=2), np.concatenate((hfsep_labels, noise_labels), axis=0), csp_title],
    [np.concatenate((np.stack((ccar_filt_hfSEP_0, ccar_filt_hfSEP_1, ccar_filt_hfSEP_2), axis=0), np.stack((ccar_filt_noise_0, ccar_filt_noise_1, ccar_filt_noise_2), axis=0)), axis=2), np.concatenate((hfsep_labels, noise_labels), axis=0), ccar_title],
    [np.concatenate((hil_extracted_CSP_hfSEP, hil_extracted_CSP_noise), axis=2), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_csp_title],
    [np.concatenate((hil_extracted_ccar_hfSEP, hil_extracted_ccar_noise), axis=2), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_ccar_title]
  ]

## Some model definition
def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
    advised to do some model searching to get optimal performance on your
    particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = keras.Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = layers.Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (1, Chans, Samples),
                                   use_bias = False)(input1)
    block1       = layers.BatchNormalization(axis = 1)(block1)
    block1       = layers.DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = tf.keras.constraints.max_norm(1.))(block1)
    block1       = layers.BatchNormalization(axis = 1)(block1)
    block1       = layers.Activation('elu')(block1)
    block1       = layers.AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = layers.SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = layers.BatchNormalization(axis = 1)(block2)
    block2       = layers.Activation('elu')(block2)
    block2       = layers.AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = layers.Flatten(name = 'flatten')(block2)
    
    dense        = layers.Dense(1, name = 'dense', 
                         kernel_constraint = tf.keras.constraints.max_norm(norm_rate))(flatten)
    sigmoid      = layers.Activation('sigmoid', name = 'sigmoid')(dense)
    
    eeg_net = keras.Model(inputs=input1, outputs=sigmoid)

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

    eeg_net.compile(optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9, clipvalue=5.0),  
      loss=tf.keras.losses.BinaryCrossentropy(), 
      metrics=METRICS)

    return eeg_net


workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_dat, noise_dat, title, idx)

for data, labels, run_title in workload:
  print('run_title: %s' % run_title)
  print(data.shape)

  ### Shuffle and split data // .T is required to switch back to shape of (trial x feature)
  shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat)
  print(shuffled_data.shape)
  X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat)
  print(X_train.shape)
  X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.33, random_state=rand_stat)

  X_train = np.expand_dims(np.swapaxes(X_train, 1, 2), axis=-1)
  X_test = np.expand_dims(np.swapaxes(X_test, 1, 2), axis=-1)
  X_eval = np.expand_dims(np.swapaxes(X_eval, 1, 2), axis=-1)

  y_train = y_train.reshape((-1,1))
  y_test = y_test.reshape((-1,1))
  y_eval = y_eval.reshape((-1,1))

  # some model definition and training
  model = EEGNet(2, X_train.shape[1], X_train.shape[2])

  history = model.fit(x=[np.squeeze(x, axis=1) for x in np.split(X_train, indices_or_sections=X_train.shape[1], axis=1)], y=y_train, epochs=25, batch_size=32, validation_data=[[np.squeeze(x, axis=1) for x in np.split(X_eval, indices_or_sections=X_eval.shape[1], axis=1)], y_eval]) 

  model.save('/media/christoph/Volume/Masterthesis/2d_tsc_november/eeg_net/models/%d/eeg_net_on_%s' % (idx, run_title.replace('/', '-')))
  np.save('/media/christoph/Volume/Masterthesis/2d_tsc_november/eeg_net/history/%d/eeg_net_on_%s' % (idx, run_title.replace('/', '-')), history.history)

  predictions = np.abs(np.rint(model.predict(x=[np.squeeze(x, axis=1) for x in np.split(X_test, indices_or_sections=X_test.shape[1], axis=1)])))
  confusion_matrix_for_model = confusion_matrix(y_test, predictions)
  tn, fp, fn, tp = confusion_matrix_for_model.ravel()
  np.save('/media/christoph/Volume/Masterthesis/2d_tsc_november/eeg_net/%d/eeg_net_on_%s_confusion_matrix' % (idx, run_title.replace('/', '-')), confusion_matrix_for_model)

  metrics = metrics_for_conf_mat(tn, fp, fn, tp)
  np.save('/media/christoph/Volume/Masterthesis/2d_tsc_november/eeg_net/%d/eeg_net_on_%s_metrics' % (idx, run_title.replace('/', '-')), metrics)
