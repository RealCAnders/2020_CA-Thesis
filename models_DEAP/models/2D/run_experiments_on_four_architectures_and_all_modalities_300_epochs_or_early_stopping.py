from __future__ import print_function 
from numpy import load                                                                                                                                              
import numpy as np                                                                                                                                                  
from os import listdir 
import numpy as np 
from numpy import load 
import sys 

from scipy.io import arff 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

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


def calculate_hil_features(transformed):
  hil_dat = scipy.signal.hilbert(transformed, axis=0)
  real_hil_dat = np.real(hil_dat)
  imag_hil_dat = np.imag(hil_dat)
  abs_hil_dat = np.abs(hil_dat)
  angle_hil_dat = np.angle(hil_dat)
  return np.stack((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat), axis=0)


modalities = [
  'epoched_data.npy',
  'epoched_filt_under_50_kx.npy',
  'epoched_filt_under_100_kx.npy',
  'epoched_filt_over_100_kx.npy',
  'epoched_filt_over_200_kx.npy'
]


starter_id = 1
end_id_so_far_loaded = 23
shape_of_data_of_one_subject = (32, 5120, 40)
def load_data_from_modality_for_all_but_to_leave_out(modality_id, identifier_to_leave_out):
  all_subjects_data = np.ones(shape_of_data_of_one_subject)

  for i in range(starter_id, end_id_so_far_loaded):
    if i is not identifier_to_leave_out:
      all_subjects_data = np.concatenate((all_subjects_data, load('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/%s' % (i, modalities[modality_id]), allow_pickle=True)), axis=2)

  return all_subjects_data[:,:,40:]



def load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(X_train, y_train, title_of_run):
  # load data in format of (channel x epoch_length x number of epochs) // all_subjects_epoched_data.shape == (32, 5120, 840)
  title_of_run = title % (identifier_to_leave_out)
  all_subjects_epoched_data = load_data_from_modality_for_all_but_to_leave_out(0, 1)

  # load specific class labels per participant, defined w threshold 5.0, (subject, trial, class) // labels_per_participant.shape == (32, 40, 4)
  labels_per_participant = np.loadtxt(open("/media/christoph/Volume/Masterthesis/DEAP_TSC/DEAP_labels_per_participant.csv", "rb"), delimiter=",")
  labels_per_participant = labels_per_participant.reshape(32, 40, 4)

  # get interesting labels; id_to_leave_out - 1 as we need to make sure to stay in range(0,...)
  labels_were_interested_in = np.asarray(np.concatenate(([labels_per_participant[x, :, :] for i, x in enumerate(range(0, 22)) if i!=id_to_leave_out - 1]), axis=0))

  all_elements_of_class_to_predict = all_subjects_epoched_data[:,:,labels_were_interested_in[:,class_to_predict] == 1]
  all_elements_not_of_class_to_predict = all_subjects_epoched_data[:,:,labels_were_interested_in[:,class_to_predict] == 0]
  ### 01_ToDo: Modify all the 11 base-modalities through: [SSD Y/N] (load respective modifiers therefore)
  # Still will remain open. Only thing to do here however: Load non-epoched data, compute SSD, epoch data
  raw_title = title_of_run + '_raw'

  ### 02_ToDo: Modify all the 22 modalities through: [leave, CSP, CCAr, bCSTP]
  # Compute CSP
  csp_title = title_of_run + '_CSP'
  csp_filters, csp_eigenvals = meet.spatfilt.CSP(all_elements_of_class_to_predict[:32,:,:].mean(2), all_elements_not_of_class_to_predict[:32,:,:].mean(2))
  all_elements_of_class_to_predict_CSP_0 = np.tensordot(csp_filters[0].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
  all_elements_not_of_class_to_predict_CSP_0 = np.tensordot(csp_filters[0].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))
  all_elements_of_class_to_predict_CSP_1 = np.tensordot(csp_filters[1].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
  all_elements_not_of_class_to_predict_CSP_1 = np.tensordot(csp_filters[1].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))
  all_elements_of_class_to_predict_CSP_2 = np.tensordot(csp_filters[2].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
  all_elements_not_of_class_to_predict_CSP_2 = np.tensordot(csp_filters[2].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))

  # Compute CCAr
  ccar_title = title_of_run + '_CCAr'
  a_ccar, b_ccar, s_ccar = meet.spatfilt.CCAvReg(all_elements_of_class_to_predict[:32,:,:])
  ccar_filt_all_elements_of_class_to_predict_0 = np.tensordot(a_ccar[:,0], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
  ccar_filt_all_elements_not_of_class_to_predict_0 = np.tensordot(a_ccar[:,0], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))
  ccar_filt_all_elements_of_class_to_predict_1 = np.tensordot(a_ccar[:,1], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
  ccar_filt_all_elements_not_of_class_to_predict_1 = np.tensordot(a_ccar[:,1], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))
  ccar_filt_all_elements_of_class_to_predict_2 = np.tensordot(a_ccar[:,2], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
  ccar_filt_all_elements_not_of_class_to_predict_2 = np.tensordot(a_ccar[:,2], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))

  # Compute bCSTP
  # s_bcstp_eigenvals, t_bcstp_eigenvals, W_bcstp, V_bcstp = bCSTP(all_elements_of_class_to_predict[:32,:,:], all_elements_not_of_class_to_predict[:32,:,:], num_iter=15, t_keep=3, s_keep=3)
  # left out as it would also need intrplt.data.... the scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_kx_data_combined_all_elements_of_class_to_predict[-1][:,0], intrplt_kx_data_combined[:8]), V_out_epoched_intrplt_kx_data_combined_all_elements_of_class_to_predict[-1][:,0][::-1], axis=-1)
  
  ### 03_ToDo: Modify all the 88 modalities through: [hil Y/N]
  hil_csp_title = title_of_run + '_CSP_hil'
  hil_extracted_all_elements_of_class_to_predict_CSP_0 = calculate_hil_features(all_elements_of_class_to_predict_CSP_0)
  hil_extracted_all_elements_not_of_class_to_predict_CSP_0 = calculate_hil_features(all_elements_not_of_class_to_predict_CSP_0)
  hil_extracted_all_elements_of_class_to_predict_CSP_1 = calculate_hil_features(all_elements_of_class_to_predict_CSP_1)
  hil_extracted_all_elements_not_of_class_to_predict_CSP_1 = calculate_hil_features(all_elements_not_of_class_to_predict_CSP_1)
  hil_extracted_all_elements_of_class_to_predict_CSP_2 = calculate_hil_features(all_elements_of_class_to_predict_CSP_2)
  hil_extracted_all_elements_not_of_class_to_predict_CSP_2 = calculate_hil_features(all_elements_not_of_class_to_predict_CSP_2)
  hil_extracted_CSP_all_elements_of_class_to_predict = np.concatenate((hil_extracted_all_elements_of_class_to_predict_CSP_0, hil_extracted_all_elements_of_class_to_predict_CSP_1, hil_extracted_all_elements_of_class_to_predict_CSP_2), axis=0)
  hil_extracted_CSP_all_elements_not_of_class_to_predict = np.concatenate((hil_extracted_all_elements_not_of_class_to_predict_CSP_0, hil_extracted_all_elements_not_of_class_to_predict_CSP_1, hil_extracted_all_elements_not_of_class_to_predict_CSP_2), axis=0)

  hil_ccar_title = title_of_run + '_CCAR_hil'
  hil_extracted_ccar_filt_all_elements_of_class_to_predict_0 = calculate_hil_features(ccar_filt_all_elements_of_class_to_predict_0)
  hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_0 = calculate_hil_features(ccar_filt_all_elements_not_of_class_to_predict_0)
  hil_extracted_ccar_filt_all_elements_of_class_to_predict_1 = calculate_hil_features(ccar_filt_all_elements_of_class_to_predict_1)
  hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_1 = calculate_hil_features(ccar_filt_all_elements_not_of_class_to_predict_1)
  hil_extracted_ccar_filt_all_elements_of_class_to_predict_2 = calculate_hil_features(ccar_filt_all_elements_of_class_to_predict_2)
  hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_2 = calculate_hil_features(ccar_filt_all_elements_not_of_class_to_predict_2)
  hil_extracted_ccar_all_elements_of_class_to_predict = np.concatenate((hil_extracted_ccar_filt_all_elements_of_class_to_predict_0, hil_extracted_ccar_filt_all_elements_of_class_to_predict_1, hil_extracted_ccar_filt_all_elements_of_class_to_predict_2), axis=0)
  hil_extracted_ccar_all_elements_not_of_class_to_predict = np.concatenate((hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_0, hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_1, hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_2), axis=0)

  all_elements_of_class_to_predict_labels = np.ones(len(all_elements_of_class_to_predict[0,0,:]), dtype=np.int8)
  all_elements_not_of_class_to_predict_labels = np.zeros(len(all_elements_not_of_class_to_predict[0,0,:]), dtype=np.int8)

  # return the datasets in epoch, channel, time_in_channel - fashion
  return [
    [np.concatenate((all_elements_of_class_to_predict, all_elements_not_of_class_to_predict), axis=-1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), raw_title],
#   [np.concatenate((hfSEP.reshape(-1, hfSEP.shape[-1]), noise.reshape(-1, noise.shape[-1])), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), title_of_run + '_all_channels_flattened'],// I have no idea why, but this singular error still occurs
    [np.concatenate((np.stack((all_elements_of_class_to_predict_CSP_0, all_elements_of_class_to_predict_CSP_1, all_elements_of_class_to_predict_CSP_2), axis=0), np.stack((ccar_filt_all_elements_not_of_class_to_predict_CSP_0, ccar_filt_all_elements_not_of_class_to_predict_CSP_1, ccar_filt_all_elements_not_of_class_to_predict_CSP_2), axis=0)), axis=2), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), csp_title],
    [np.concatenate((np.stack((ccar_filt_all_elements_of_class_to_predict_0, ccar_filt_all_elements_of_class_to_predict_1, ccar_filt_all_elements_of_class_to_predict_2), axis=0), np.stack((ccar_filt_ccar_filt_all_elements_not_of_class_to_predict_0, ccar_filt_ccar_filt_all_elements_not_of_class_to_predict_1, ccar_filt_ccar_filt_all_elements_not_of_class_to_predict_2), axis=0)), axis=2), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), ccar_title],
    [np.concatenate((hil_extracted_CSP_all_elements_of_class_to_predict, hil_extracted_CSP_all_elements_not_of_class_to_predict), axis=2), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), hil_csp_title],
    [np.concatenate((hil_extracted_ccar_all_elements_of_class_to_predict, hil_extracted_ccar_all_elements_not_of_class_to_predict), axis=2), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), hil_ccar_title]
  ]


######
### Begin Model definitions in order: MODEL_BLKOCK_0[adjusted, original], ... MODEL_BLOCK_n 
######

# define early-stopping to be used for all models
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_binary_accuracy', min_delta=0, patience=25, verbose=0, mode='max',
    baseline=None, restore_best_weights=True
)

### DenseConvNet
def create_DenseConvNet_for_2d_adjusted(input_len, amount_of_channels):

  # Define the respective channels for the electrodes signals
  model = keras.models.Sequential([  
    keras.layers.Conv2D(filters=25, kernel_size=(amount_of_channels,1), input_shape=(amount_of_channels,input_len,1)),  
    keras.layers.Conv2D(filters=25, kernel_size=(1,20), activation='elu'),  
    keras.layers.MaxPool2D(pool_size=(1,3), strides=(3,1)), 
    keras.layers.Conv2D(filters=50, kernel_size=(1,10), strides=(1,1), activation='elu', padding="valid"), 
    keras.layers.MaxPool2D(pool_size=(1,3), strides=(3,1)), 
    keras.layers.Conv2D(filters=100, kernel_size=(1,8), strides=(1,1), activation='elu', padding="valid"), 
    keras.layers.MaxPool2D(pool_size=(1,3), strides=(3,1)), 
    keras.layers.Conv2D(filters=200, kernel_size=(1,2), strides=(1,1), activation='elu', padding="valid"), 
    keras.layers.MaxPool2D(pool_size=(1,3), strides=(3,1)), 
    keras.layers.Flatten(),
    keras.layers.Activation('softmax', name = 'softmax'),
    keras.layers.Dense(1, activation='sigmoid')  
  ]) 

  model.summary()

  METRICS = [
    tf.keras.metrics.TruePositives(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.TrueNegatives(),
    tf.keras.metrics.FalseNegatives(), 
    tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.AUC(),
    tf.keras.metrics.MeanAbsoluteError(),
  ]

  model.compile(optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9, clipvalue=5.0),  
    loss=tf.keras.losses.BinaryCrossentropy(), 
    metrics=METRICS)

  return model


def create_DenseConvNet_for_2d_original(input_len, amount_of_channels):

  # Define the respective channels for the electrodes signals
  model = keras.models.Sequential([  
    keras.layers.Conv2D(filters=25, kernel_size=(amount_of_channels,1), input_shape=(amount_of_channels,input_len,1)),  
    keras.layers.Conv2D(filters=25, kernel_size=(1,44), activation='elu'),  
    keras.layers.MaxPool2D(pool_size=(1,3), strides=(3,1)), 
    keras.layers.Conv2D(filters=50, kernel_size=(1,25), strides=(1,1), activation='elu', padding="valid"), 
    keras.layers.MaxPool2D(pool_size=(1,3), strides=(3,1)), 
    keras.layers.Conv2D(filters=100, kernel_size=(1,50), strides=(1,1), activation='elu', padding="valid"), 
    keras.layers.MaxPool2D(pool_size=(1,3), strides=(3,1)), 
    keras.layers.Conv2D(filters=200, kernel_size=(1,100), strides=(1,1), activation='elu', padding="valid"), 
    keras.layers.MaxPool2D(pool_size=(1,3), strides=(3,1)), 
    keras.layers.Flatten(),
    keras.layers.Activation('softmax', name = 'softmax'),
    keras.layers.Dense(1, activation='sigmoid')  
  ]) 

  model.summary()

  METRICS = [
    tf.keras.metrics.TruePositives(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.TrueNegatives(),
    tf.keras.metrics.FalseNegatives(), 
    tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.AUC(),
    tf.keras.metrics.MeanAbsoluteError(),
  ]

  model.compile(optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9, clipvalue=5.0),  
    loss=tf.keras.losses.BinaryCrossentropy(), 
    metrics=METRICS)

  return model 


### ShallowConvNet
def create_shallow_conv_net_for_2d_adjusted(input_len, amount_of_channels):

  # Define the respective channels for the electrodes signals
  model = keras.models.Sequential([  
      keras.layers.Conv2D(filters=40, kernel_size=(1,10), input_shape=(amount_of_channels,input_len,1)),  
      keras.layers.Conv2D(filters=40, kernel_size=(amount_of_channels,40), activation='elu'),  
      keras.layers.AveragePooling2D(pool_size=(1,2), strides=(15,1)),
      keras.layers.Flatten(),
      keras.layers.Activation('softmax', name = 'softmax'),
      keras.layers.Dense(1, activation='sigmoid')  
  ])

  model.summary()

  METRICS = [
    tf.keras.metrics.TruePositives(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.TrueNegatives(),
    tf.keras.metrics.FalseNegatives(), 
    tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.AUC(),
    tf.keras.metrics.MeanAbsoluteError(),
  ]

  model.compile(optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9, clipvalue=5.0),  
    loss=tf.keras.losses.BinaryCrossentropy(), 
    metrics=METRICS)

  return model


def create_shallow_conv_net_for_2d_original(input_len, amount_of_channels):

  # Define the respective channels for the electrodes signals
  model = keras.models.Sequential([  
      keras.layers.Conv2D(filters=40, kernel_size=(1,10), input_shape=(amount_of_channels,input_len,1)),  
      keras.layers.Conv2D(filters=40, kernel_size=(amount_of_channels,40), activation='elu'),  
      keras.layers.AveragePooling2D(pool_size=(1,75), strides=(15,1)),
      keras.layers.Flatten(),
      keras.layers.Activation('softmax', name = 'softmax'),
      keras.layers.Dense(1, activation='sigmoid')  
  ])

  model.summary()

  METRICS = [
    tf.keras.metrics.TruePositives(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.TrueNegatives(),
    tf.keras.metrics.FalseNegatives(), 
    tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.AUC(),
    tf.keras.metrics.MeanAbsoluteError(),
  ]

  model.compile(optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9, clipvalue=5.0),  
    loss=tf.keras.losses.BinaryCrossentropy(), 
    metrics=METRICS)

  return model


### MC-CNN
def create_mc_cnn_for_2d_adjusted(input_len, amount_of_channels):

  # Define the respective channels for the electrodes signals
  channel_models = []
  input_channels = []
  for channel_id in range(amount_of_channels):
    input_channel = tf.keras.Input((input_len, 1), name=('Channel-%d' % channel_id))
    c_conv = tf.keras.layers.Conv1D(32, 3, use_bias=False)(input_channel)
    c_pool = tf.keras.layers.AveragePooling1D(pool_size=2)(c_conv)
    c_conv = tf.keras.layers.Conv1D(16, 4, use_bias=False)(c_pool)
    c_pool = tf.keras.layers.MaxPool1D(pool_size=2)(c_conv)
    out = tf.keras.layers.Activation('sigmoid')(c_pool)
    channel_model = tf.keras.Model(inputs=input_channel, outputs=out)
  
    input_channels.append(input_channel)    
    channel_models.append(channel_model)
  
  # concatente the 1D-channels to a MC-Model
  combined_model = tf.keras.layers.concatenate(
      [tf.keras.backend.expand_dims(channl.output, axis=1) for channl in channel_models]
      , axis=1
  )
  
  # further define the MC-Model using Conv2D, Dense, Pooling and Flattening
  conv = tf.keras.layers.Conv2D(128, (2, 5))(combined_model)
  den = tf.keras.layers.Dense(512)(conv)
  conv = tf.keras.layers.Conv2D(128, (2, 3))(den)
  den = tf.keras.layers.Dense(512)(conv)
  conv = tf.keras.layers.Conv2D(64, (1, 2))(den)
  avg = tf.keras.layers.AveragePooling2D((1, 2))(conv)
  flat = tf.keras.layers.Flatten()(avg)
  drop = tf.keras.layers.Dropout(0.5)(flat)
  den = tf.keras.layers.Dense(10)(drop)
  out = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(den)
  
  # Build the final model, show a summary and compile it w. metrics
  mc_cnn = tf.keras.Model(inputs=input_channels, outputs=out)

  mc_cnn.summary()

  METRICS = [
    tf.keras.metrics.TruePositives(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.TrueNegatives(),
    tf.keras.metrics.FalseNegatives(), 
    tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.AUC(),
    tf.keras.metrics.MeanAbsoluteError(),
  ]

  mc_cnn.compile(optimizer=tf.keras.optimizers.SGD(0.0001, momentum=0.9, clipvalue=5.0),  
    loss=tf.keras.losses.BinaryCrossentropy(), 
    metrics=METRICS)

  return mc_cnn


def create_mc_cnn_for_2d_original(input_len, amount_of_channels):

  # Define the respective channels for the electrodes signals
  channel_models = []
  input_channels = []
  for channel_id in range(amount_of_channels):
    input_channel = tf.keras.Input((input_len, 1), name=('Channel-%d' % channel_id))
    c_conv = tf.keras.layers.Conv1D(32, 3, use_bias=False)(input_channel)
    c_pool = tf.keras.layers.AveragePooling1D(pool_size=2)(c_conv)
    c_conv = tf.keras.layers.Conv1D(16, 4, use_bias=False)(c_pool)
    c_pool = tf.keras.layers.MaxPool1D(pool_size=2)(c_conv)
    out = tf.keras.layers.Activation('sigmoid')(c_pool)
    channel_model = tf.keras.Model(inputs=input_channel, outputs=out)
  
    input_channels.append(input_channel)    
    channel_models.append(channel_model)
  
  # concatente the 1D-channels to a MC-Model
  combined_model = tf.keras.layers.concatenate(
      [tf.keras.backend.expand_dims(channl.output, axis=1) for channl in channel_models]
      , axis=1
  )
  
  # further define the MC-Model using Conv2D, Dense, Pooling and Flattening
  conv = tf.keras.layers.Conv2D(128, (2, 5))(combined_model)
  den = tf.keras.layers.Dense(128)(conv)
  conv = tf.keras.layers.Conv2D(128, (2, 15))(den)
  den = tf.keras.layers.Dense(128)(conv)
  conv = tf.keras.layers.Conv2D(64, (1, 10))(den)
  avg = tf.keras.layers.AveragePooling2D((1, 8))(conv)
  flat = tf.keras.layers.Flatten()(avg)
  drop = tf.keras.layers.Dropout(0.5)(flat)
  den = tf.keras.layers.Dense(10)(drop)
  out = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(den)
  
  # Build the final model, show a summary and compile it w. metrics
  mc_cnn = tf.keras.Model(inputs=input_channels, outputs=out)

  mc_cnn.summary()

  METRICS = [
    tf.keras.metrics.TruePositives(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.TrueNegatives(),
    tf.keras.metrics.FalseNegatives(), 
    tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.AUC(),
    tf.keras.metrics.MeanAbsoluteError(),
  ]

  mc_cnn.compile(optimizer=tf.keras.optimizers.SGD(0.0001, momentum=0.9, clipvalue=5.0),  
    loss=tf.keras.losses.BinaryCrossentropy(), 
    metrics=METRICS)

  return mc_cnn


### EEGNet
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
      tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
      tf.keras.metrics.Precision(),
      tf.keras.metrics.Recall(),
      tf.keras.metrics.AUC(),
      tf.keras.metrics.MeanAbsoluteError(),
    ]

    eeg_net.compile(optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9, clipvalue=5.0),  
      loss=tf.keras.losses.BinaryCrossentropy(), 
      metrics=METRICS)

    return eeg_net


######
### End Model definitions in order: MODEL_BLKOCK_0[adjusted, original], ... MODEL_BLOCK_n 
######


train_identifier = 'TRAIN'
test_identifier = 'TEST'
num_epochs = 300
rand_stat = 42

# ToDo: Adjust for final run!
datasets_ucr = [
  ['/home/christoph/Downloads/Thesis_Last_Minute_TSC/FingerMovements/FingerMovements_%s.arff', "b'left'"],
#  ['/home/christoph/Downloads/Thesis_Last_Minute_TSC/FaceDetection/FaceDetection_%s.arff', "b'0'"],
  ['/home/christoph/Downloads/Thesis_Last_Minute_TSC/MotorImagery/MotorImagery_%s.arff', "b'tongue'"],
  ['/home/christoph/Downloads/Thesis_Last_Minute_TSC/SelfRegulationSCP1/SelfRegulationSCP1_%s.arff', "b'negativity"],
  ['/home/christoph/Downloads/Thesis_Last_Minute_TSC/SelfRegulationSCP2/SelfRegulationSCP2_%s.arff', "b'negativity"],
]


for idx, ucr_tuple in enumerate(datasets_ucr):
  X_train, y_train = load_data_from_ucr_set(ucr_tuple[0], ucr_tuple[1], train_identifier)
  X_test, y_test = load_data_from_ucr_set(ucr_tuple[0], ucr_tuple[1], test_identifier)

  print('Dataset %s has %d train cases and %d test cases.' % (ucr_tuple[0].split('/')[-1].split('_')[0], X_train.shape[-1], X_test.shape[-1]))
  print('Following are shapes of train and test')
  print(X_train.shape)
  print(X_test.shape)

  workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(X_train, y_train, ucr_tuple[0].split('/')[-1].split('_')[0])
  workload_test = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(X_test, y_test, ucr_tuple[0].split('/')[-1].split('_')[0])

  ctr = 0
  for data, labels, run_title in workload:
    print('run_title: %s' % run_title)
    print(data.shape)

    ### Shuffle and split data // .T is required to switch back to shape of (trial x feature)
    shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat)
    print(shuffled_data.shape)
    X_train, X_eval, y_train, y_eval = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat)
    print(X_train.shape)

    # results in shape: (chan x sample_time x trial x 1):e.g.:(7, 200, 1152, 1)
    X_train = np.expand_dims(np.swapaxes(X_train, 1, 2), axis=-1)
    X_test = np.swapaxes(np.expand_dims(np.swapaxes(workload_test[ctr][0], 1, 2), axis=-1), 0, 1)
    X_eval = np.expand_dims(np.swapaxes(X_eval, 1, 2), axis=-1)

    y_train = y_train.reshape((-1,1))
    y_test = workload_test[ctr][1].reshape((-1,1))
    y_eval = y_eval.reshape((-1,1))

    if idx <= -1:
      # For the datasets FaceDetection and FingerMovements the models had to be adjusted
      models = [
        [create_DenseConvNet_for_2d_adjusted(X_train.shape[2], X_train.shape[1]), 'dense_conv_net'],
        [create_shallow_conv_net_for_2d_adjusted(X_train.shape[2], X_train.shape[1]), 'shallow_conv_net'],
        [EEGNet(2, X_train.shape[1], X_train.shape[2]), 'eeg_net'],
        [create_mc_cnn_for_2d_adjusted(X_train.shape[2], X_train.shape[1]), 'mc_cnn']
      ]
      
      for model_tuple in models:
        model = model_tuple[0]
        if model_tuple[1] in 'mc_cnn':  
          history = model.fit(
            x=[np.squeeze(x, axis=1) for x in np.split(X_train, indices_or_sections=X_train.shape[1], axis=1)], y=y_train, 
            epochs=num_epochs, batch_size=32, 
            validation_data=[[np.squeeze(x, axis=1) for x in np.split(X_eval, indices_or_sections=X_eval.shape[1], axis=1)], y_eval],
            callbacks=[callback]
          )
        else:
          history = model.fit(x=X_train, y=y_train, epochs=num_epochs, batch_size=32, validation_data=[X_eval, y_eval], callbacks=[callback])

        model.save('/media/christoph/Volume/Masterthesis/UCR_TSC/scripts_and_results/early_stopping_300_epochs_or_convergence/%s/models/%s_on_%s' % (model_tuple[1], model_tuple[1], run_title.replace('/', '-')))
        np.save('/media/christoph/Volume/Masterthesis/UCR_TSC/scripts_and_results/early_stopping_300_epochs_or_convergence/%s/histories/%s_on_%s' % (model_tuple[1], model_tuple[1], run_title.replace('/', '-')), history.history)
      
        if model_tuple[1] in 'mc_cnn':
          predictions = np.abs(np.rint(model.predict(x=[np.squeeze(x, axis=1) for x in np.split(X_test, indices_or_sections=X_test.shape[1], axis=1)])))
        else:
          predictions = np.abs(np.rint(model.predict(x=X_test)))
        
        confusion_matrix_for_model = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = confusion_matrix_for_model.ravel()
        np.save('/media/christoph/Volume/Masterthesis/UCR_TSC/scripts_and_results/early_stopping_300_epochs_or_convergence/%s/metrics/%s_on_%s_confusion_matrix' % (model_tuple[1], model_tuple[1], run_title.replace('/', '-')), confusion_matrix_for_model)
      
        metrics = metrics_for_conf_mat(tn, fp, fn, tp)
        np.save('/media/christoph/Volume/Masterthesis/UCR_TSC/scripts_and_results/early_stopping_300_epochs_or_convergence/%s/metrics/%s_on_%s_metrics' % (model_tuple[1], model_tuple[1], run_title.replace('/', '-')), metrics)

    else:
      # For the datasets FaceDetection and FingerMovements the models had to be adjusted
      models = [
        [create_DenseConvNet_for_2d_original(X_train.shape[2], X_train.shape[1]), 'dense_conv_net'],
        [create_shallow_conv_net_for_2d_original(X_train.shape[2], X_train.shape[1]), 'shallow_conv_net'],
        [EEGNet(2, X_train.shape[1], X_train.shape[2]), 'eeg_net'],
        [create_mc_cnn_for_2d_original(X_train.shape[2], X_train.shape[1]), 'mc_cnn']
      ]
      
      for model_tuple in models:
        model = model_tuple[0]
        if model_tuple[1] in 'mc_cnn':  
          history = model.fit(
            x=[np.squeeze(x, axis=1) for x in np.split(X_train, indices_or_sections=X_train.shape[1], axis=1)], y=y_train, 
            epochs=num_epochs, batch_size=32, 
            validation_data=[[np.squeeze(x, axis=1) for x in np.split(X_eval, indices_or_sections=X_eval.shape[1], axis=1)], y_eval],
            callbacks=[callback]
          )
        else:
          history = model.fit(x=X_train, y=y_train, epochs=num_epochs, batch_size=32, validation_data=[X_eval, y_eval], callbacks=[callback])

        model.save('/media/christoph/Volume/Masterthesis/UCR_TSC/scripts_and_results/early_stopping_300_epochs_or_convergence/%s/models/%s_on_%s' % (model_tuple[1], model_tuple[1], run_title.replace('/', '-')))
        np.save('/media/christoph/Volume/Masterthesis/UCR_TSC/scripts_and_results/early_stopping_300_epochs_or_convergence/%s/histories/%s_on_%s' % (model_tuple[1], model_tuple[1], run_title.replace('/', '-')), history.history)
      
        if model_tuple[1] in 'mc_cnn':
          predictions = np.abs(np.rint(model.predict(x=[np.squeeze(x, axis=1) for x in np.split(X_test, indices_or_sections=X_test.shape[1], axis=1)])))
        else:
          predictions = np.abs(np.rint(model.predict(x=X_test)))
        
        confusion_matrix_for_model = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = confusion_matrix_for_model.ravel()
        np.save('/media/christoph/Volume/Masterthesis/UCR_TSC/scripts_and_results/early_stopping_300_epochs_or_convergence/%s/metrics/%s_on_%s_confusion_matrix' % (model_tuple[1], model_tuple[1], run_title.replace('/', '-')), confusion_matrix_for_model)
      
        metrics = metrics_for_conf_mat(tn, fp, fn, tp)
        np.save('/media/christoph/Volume/Masterthesis/UCR_TSC/scripts_and_results/early_stopping_300_epochs_or_convergence/%s/metrics/%s_on_%s_metrics' % (model_tuple[1], model_tuple[1], run_title.replace('/', '-')), metrics)

    ctr += 1
    print('Change in counter-value from [%d] to [%d]' % (ctr-1, ctr))


#-#-#-#    # some model definition and training
#-#-#-#    model = create_DenseConvNet_for_2d(X_train.shape[1], X_train.shape[0])

#-#-#-#    # toDo: Add training with data derived from _ctr_
#-#-#-#    #history = model.fit(x=X_train, y=y_train, epochs=25, batch_size=32, validation_data=[X_eval, y_eval])
#-#-#-#    #history = model.fit(x=[np.squeeze(x, axis=1) for x in np.split(X_train, indices_or_sections=X_train.shape[1], axis=1)], y=y_train, epochs=25, batch_size=32, validation_data=[[np.squeeze(x, axis=1) for x in np.split(X_eval, indices_or_sections=X_eval.shape[1], axis=1)], y_eval]) 

#-#-#-#    model.save('/media/christoph/Volume/Masterthesis/multi-channel-training/dense_conv_net/models/%d/dense_conv_net_on%s' % (idx, run_title.replace('/', '-')))
#-#-#-#    np.save('/media/christoph/Volume/Masterthesis/multi-channel-training/dense_conv_net/history/%d/dense_conv_net_on%s' % (idx, run_title.replace('/', '-')), history.history)
  
#-#-#-#    #predictions = np.abs(np.rint(model.predict(x=[np.squeeze(x, axis=1) for x in np.split(X_test, indices_or_sections=X_test.shape[1], axis=1)])))
#-#-#-#    predictions = np.abs(np.rint(model.predict(x=X_test)))
#-#-#-#    confusion_matrix_for_model = confusion_matrix(y_test, predictions)
#-#-#-#    tn, fp, fn, tp = confusion_matrix_for_model.ravel()
#-#-#-#    np.save('/media/christoph/Volume/Masterthesis/multi-channel-training/dense_conv_net/%d/dense_conv_net_on%s_confusion_matrix' % (idx, run_title.replace('/', '-')), confusion_matrix_for_model)
  
#-#-#-#    metrics = metrics_for_conf_mat(tn, fp, fn, tp)
#-#-#-#    np.save('/media/christoph/Volume/Masterthesis/multi-channel-training/dense_conv_net/%d/dense_conv_net_on%s_metrics' % (idx, run_title.replace('/', '-')), metrics)