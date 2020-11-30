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


args = sys.argv
print('Number of arguments: %d arguments.' % len(args))
print('Argument List:', str(args))

rand_stat = 42
id_to_leave_out = int(sys.argv[1])


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


modalities = [
  'epoched_data.npy',
  'epoched_filt_under_50_kx.npy',
  'epoched_filt_under_100_kx.npy',
  'epoched_filt_over_100_kx.npy',
]


starter_id = 1
end_id_so_far_loaded = 23
shape_of_data_of_one_subject = (32, 2560, 40)
def load_data_from_modality_for_all_but_to_leave_out(modality_id, identifier_to_leave_out):
  all_subjects_data = np.ones(shape_of_data_of_one_subject)

  for i in range(starter_id, end_id_so_far_loaded):
    if i is not identifier_to_leave_out:
      new_to_add_data = load('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/%s' % (i, modality_id), allow_pickle=True)[:32,:,:]
      all_subjects_data = np.concatenate((all_subjects_data, new_to_add_data), axis=2)
      print(new_to_add_data.shape)

  return all_subjects_data[:,:,40:]


def load_data_from_modality_for_one_to_leave_out(modality_id, identifier_to_leave_out):
  all_subjects_data = np.ones(shape_of_data_of_one_subject)

  for i in range(starter_id, end_id_so_far_loaded):
    if i is identifier_to_leave_out:
      all_subjects_data = np.concatenate((all_subjects_data, load('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/%s' % (i, modality_id), allow_pickle=True)[:32,:,:]), axis=2)

  return all_subjects_data[:,:,40:]


def calculate_hil_features(transformed):
  hil_dat = scipy.signal.hilbert(transformed, axis=0)
  real_hil_dat = np.real(hil_dat)
  imag_hil_dat = np.imag(hil_dat)
  abs_hil_dat = np.abs(hil_dat)
  angle_hil_dat = np.angle(hil_dat)
  return np.concatenate((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat), axis=0)


def load_test_data(modality_id, title, identifier_to_leave_out, class_to_predict, modifier_for_run):
  # load data in format of (channel x epoch_length x number of epochs) // all_subjects_epoched_data.shape == (32, 5120, 840)
  title_of_run = title % (identifier_to_leave_out)
  all_subjects_epoched_data = load_data_from_modality_for_one_to_leave_out(modality_id, identifier_to_leave_out)

  # load specific class labels per participant, defined w threshold 5.0, (subject, trial, class) // labels_per_participant.shape == (32, 40, 4)
  labels_for_participant = np.load('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/labels.npy' % identifier_to_leave_out, allow_pickle=True)

  overall_labels = []
  for labels in labels_for_participant:
    for j in range(5):
      overall_labels.append(labels)
  # get interesting labels; id_to_leave_out - 1 as we need to make sure to stay in range(0,...)
  labels_were_interested_in = np.asarray(overall_labels)
#  labels_were_interested_in = labels_for_participant

  all_elements_of_class_to_predict = all_subjects_epoched_data[:,:,labels_were_interested_in[:,class_to_predict] == 1]
  if all_elements_of_class_to_predict.shape[2] <= 1:
    all_elements_of_class_to_predict = all_subjects_epoched_data[:,:,:2]

  all_elements_not_of_class_to_predict = all_subjects_epoched_data[:,:,labels_were_interested_in[:,class_to_predict] == 0]
  if all_elements_not_of_class_to_predict.shape[2] <= 1:
    all_elements_not_of_class_to_predict = all_subjects_epoched_data[:,:,:2]
  ### 01_ToDo: Modify all the 11 base-modalities through: [SSD Y/N] (load respective modifiers therefore)
  # Still will remain open. Only thing to do here however: Load non-epoched data, compute SSD, epoch data
  modifier_for_run = '_' + modifier_for_run

  all_elements_of_class_to_predict_labels = np.ones(len(all_elements_of_class_to_predict[0,0,:]), dtype=np.int8)
  all_elements_not_of_class_to_predict_labels = np.zeros(len(all_elements_not_of_class_to_predict[0,0,:]), dtype=np.int8)

  raw_title = title_of_run + '_raw'
  if '_raw' in modifier_for_run:
    return [np.concatenate((all_elements_of_class_to_predict[5]-all_elements_of_class_to_predict[0], all_elements_not_of_class_to_predict[5]-all_elements_not_of_class_to_predict[0]), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), raw_title]

  ### 02_ToDo: Modify all the 22 modalities through: [leave, CSP, CCAr, bCSTP]
  # Compute CSP
  csp_title = title_of_run + '_CSP'
  if '_CSP' in modifier_for_run:
    csp_filters, csp_eigenvals = meet.spatfilt.CSP(all_elements_of_class_to_predict[:32,:,:].mean(2), all_elements_not_of_class_to_predict[:32,:,:].mean(2))
    all_elements_of_class_to_predict_CSP_0 = np.tensordot(csp_filters[0].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_not_of_class_to_predict_CSP_0 = np.tensordot(csp_filters[0].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_of_class_to_predict_CSP_1 = np.tensordot(csp_filters[1].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_not_of_class_to_predict_CSP_1 = np.tensordot(csp_filters[1].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_of_class_to_predict_CSP_2 = np.tensordot(csp_filters[2].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_not_of_class_to_predict_CSP_2 = np.tensordot(csp_filters[2].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))
    return [np.concatenate((all_elements_of_class_to_predict_CSP_0, all_elements_not_of_class_to_predict_CSP_0), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), csp_title]

  # Compute CCAr
  ccar_title = title_of_run + '_CCAr'
  if '_CCAr' in modifier_for_run:
    a_ccar, b_ccar, s_ccar = meet.spatfilt.CCAvReg(all_elements_of_class_to_predict[:32,:,:])
    ccar_filt_all_elements_of_class_to_predict_0 = np.tensordot(a_ccar[:,0], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_not_of_class_to_predict_0 = np.tensordot(a_ccar[:,0], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_of_class_to_predict_1 = np.tensordot(a_ccar[:,1], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_not_of_class_to_predict_1 = np.tensordot(a_ccar[:,1], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_of_class_to_predict_2 = np.tensordot(a_ccar[:,2], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_not_of_class_to_predict_2 = np.tensordot(a_ccar[:,2], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))
    return [np.concatenate((ccar_filt_all_elements_of_class_to_predict_0, ccar_filt_all_elements_not_of_class_to_predict_0), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), ccar_title]

  # Compute bCSTP
  # s_bcstp_eigenvals, t_bcstp_eigenvals, W_bcstp, V_bcstp = bCSTP(all_elements_of_class_to_predict[:32,:,:], all_elements_not_of_class_to_predict[:32,:,:], num_iter=15, t_keep=3, s_keep=3)
  # left out as it would also need intrplt.data.... the scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_kx_data_combined_all_elements_of_class_to_predict[-1][:,0], intrplt_kx_data_combined[:8]), V_out_epoched_intrplt_kx_data_combined_all_elements_of_class_to_predict[-1][:,0][::-1], axis=-1)
  
  ### 03_ToDo: Modify all the 88 modalities through: [hil Y/N]
  hil_csp_title = title_of_run + '_CSP_hil'
  if '_CSP_hil' in modifier_for_run:
    csp_filters, csp_eigenvals = meet.spatfilt.CSP(all_elements_of_class_to_predict[:32,:,:].mean(2), all_elements_not_of_class_to_predict[:32,:,:].mean(2))
    all_elements_of_class_to_predict_CSP_0 = np.tensordot(csp_filters[0].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_not_of_class_to_predict_CSP_0 = np.tensordot(csp_filters[0].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_of_class_to_predict_CSP_1 = np.tensordot(csp_filters[1].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_not_of_class_to_predict_CSP_1 = np.tensordot(csp_filters[1].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_of_class_to_predict_CSP_2 = np.tensordot(csp_filters[2].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_not_of_class_to_predict_CSP_2 = np.tensordot(csp_filters[2].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))

    hil_extracted_all_elements_of_class_to_predict_CSP_0 = calculate_hil_features(all_elements_of_class_to_predict_CSP_0)
    hil_extracted_all_elements_not_of_class_to_predict_CSP_0 = calculate_hil_features(all_elements_not_of_class_to_predict_CSP_0)
    hil_extracted_all_elements_of_class_to_predict_CSP_1 = calculate_hil_features(all_elements_of_class_to_predict_CSP_1)
    hil_extracted_all_elements_not_of_class_to_predict_CSP_1 = calculate_hil_features(all_elements_not_of_class_to_predict_CSP_1)
    hil_extracted_all_elements_of_class_to_predict_CSP_2 = calculate_hil_features(all_elements_of_class_to_predict_CSP_2)
    hil_extracted_all_elements_not_of_class_to_predict_CSP_2 = calculate_hil_features(all_elements_not_of_class_to_predict_CSP_2)
    hil_extracted_CSP_all_elements_of_class_to_predict = np.concatenate((hil_extracted_all_elements_of_class_to_predict_CSP_0, hil_extracted_all_elements_of_class_to_predict_CSP_1, hil_extracted_all_elements_of_class_to_predict_CSP_2), axis=0)
    hil_extracted_CSP_all_elements_not_of_class_to_predict = np.concatenate((hil_extracted_all_elements_not_of_class_to_predict_CSP_0, hil_extracted_all_elements_not_of_class_to_predict_CSP_1, hil_extracted_all_elements_not_of_class_to_predict_CSP_2), axis=0)
    return [np.concatenate((hil_extracted_CSP_all_elements_of_class_to_predict, hil_extracted_CSP_all_elements_not_of_class_to_predict), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), hil_csp_title]

  hil_ccar_title = title_of_run + '_CCAR_hil'
  if '_CCAR_hil' in modifier_for_run:
    a_ccar, b_ccar, s_ccar = meet.spatfilt.CCAvReg(all_elements_of_class_to_predict[:32,:,:])
    ccar_filt_all_elements_of_class_to_predict_0 = np.tensordot(a_ccar[:,0], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_not_of_class_to_predict_0 = np.tensordot(a_ccar[:,0], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_of_class_to_predict_1 = np.tensordot(a_ccar[:,1], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_not_of_class_to_predict_1 = np.tensordot(a_ccar[:,1], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_of_class_to_predict_2 = np.tensordot(a_ccar[:,2], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_not_of_class_to_predict_2 = np.tensordot(a_ccar[:,2], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))
    
    hil_extracted_ccar_filt_all_elements_of_class_to_predict_0 = calculate_hil_features(ccar_filt_all_elements_of_class_to_predict_0)
    hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_0 = calculate_hil_features(ccar_filt_all_elements_not_of_class_to_predict_0)
    hil_extracted_ccar_filt_all_elements_of_class_to_predict_1 = calculate_hil_features(ccar_filt_all_elements_of_class_to_predict_1)
    hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_1 = calculate_hil_features(ccar_filt_all_elements_not_of_class_to_predict_1)
    hil_extracted_ccar_filt_all_elements_of_class_to_predict_2 = calculate_hil_features(ccar_filt_all_elements_of_class_to_predict_2)
    hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_2 = calculate_hil_features(ccar_filt_all_elements_not_of_class_to_predict_2)
    hil_extracted_ccar_all_elements_of_class_to_predict = np.concatenate((hil_extracted_ccar_filt_all_elements_of_class_to_predict_0, hil_extracted_ccar_filt_all_elements_of_class_to_predict_1, hil_extracted_ccar_filt_all_elements_of_class_to_predict_2), axis=0)
    hil_extracted_ccar_all_elements_not_of_class_to_predict = np.concatenate((hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_0, hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_1, hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_2), axis=0)
    return [np.concatenate((hil_extracted_ccar_all_elements_of_class_to_predict, hil_extracted_ccar_all_elements_not_of_class_to_predict), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), hil_ccar_title]

  # return the datasets in epoch, channel, time_in_channel - fashion
  return [
    [np.concatenate((all_elements_of_class_to_predict[5]-all_elements_of_class_to_predict[0], all_elements_not_of_class_to_predict[5]-all_elements_not_of_class_to_predict[0]), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), raw_title],
#   [np.concatenate((all_elements_of_class_to_predict.reshape(-1, all_elements_of_class_to_predict.shape[-1]), all_elements_not_of_class_to_predict.reshape(-1, all_elements_not_of_class_to_predict.shape[-1])), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), title_of_run + '_all_channels_flattened'],// I have no idea why, but this singular error still occurs
    [np.concatenate((all_elements_of_class_to_predict_CSP_0, all_elements_not_of_class_to_predict_CSP_0), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), csp_title],
    [np.concatenate((ccar_filt_all_elements_of_class_to_predict_0, ccar_filt_all_elements_not_of_class_to_predict_0), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), ccar_title],
    [np.concatenate((hil_extracted_CSP_all_elements_of_class_to_predict, hil_extracted_CSP_all_elements_not_of_class_to_predict), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), hil_csp_title],
    [np.concatenate((hil_extracted_ccar_all_elements_of_class_to_predict, hil_extracted_ccar_all_elements_not_of_class_to_predict), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), hil_ccar_title]
  ]


def load_data_as_still_sorted_all_elements_of_class_to_predict_noise_data_then_labels_from_one_subject(modality_id, title, identifier_to_leave_out, class_to_predict, modifier_for_run):
  # load data in format of (channel x epoch_length x number of epochs) // all_subjects_epoched_data.shape == (32, 5120, 840)
  title_of_run = title % (identifier_to_leave_out)
  all_subjects_epoched_data = load_data_from_modality_for_all_but_to_leave_out(modality_id, identifier_to_leave_out)

  # load specific class labels per participant, defined w threshold 5.0, (subject, trial, class) // labels_per_participant.shape == (32, 40, 4)
  overall_labels = []
  for ij in range(starter_id, end_id_so_far_loaded):
     if ij != id_to_leave_out:
      labels_for_participant = np.load('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/labels.npy' % ij, allow_pickle=True)
      for labels in labels_for_participant:
#        print(labels)
        for j in range(5):
          overall_labels.append(labels)
#       overall_labels.append(labels_for_participant)

  # get interesting labels; id_to_leave_out - 1 as we need to make sure to stay in range(0,...)
  labels_were_interested_in = np.asarray(overall_labels)

  print('FINDMEEEE in modality: %s leave out %d for class %d' % (modality_id, identifier_to_leave_out, class_to_predict))
  print(all_subjects_epoched_data.shape)
  print('FINDMEEEE in modality: %s leave out %d for class %d' % (modality_id, identifier_to_leave_out, class_to_predict))
  print(labels_were_interested_in.shape)

  all_elements_of_class_to_predict = all_subjects_epoched_data[:,:,labels_were_interested_in[:,class_to_predict] == 1]
  if all_elements_of_class_to_predict.shape[2] <= 1:
    all_elements_of_class_to_predict = all_subjects_epoched_data[:,:,:2]

  all_elements_not_of_class_to_predict = all_subjects_epoched_data[:,:,labels_were_interested_in[:,class_to_predict] == 0]
  if all_elements_not_of_class_to_predict.shape[2] <= 1:
    all_elements_not_of_class_to_predict = all_subjects_epoched_data[:,:,:2]
  ### 01_ToDo: Modify all the 11 base-modalities through: [SSD Y/N] (load respective modifiers therefore)
  # Still will remain open. Only thing to do here however: Load non-epoched data, compute SSD, epoch data

  modifier_for_run = '_' + modifier_for_run

  all_elements_of_class_to_predict_labels = np.ones(len(all_elements_of_class_to_predict[0,0,:]), dtype=np.int8)
  all_elements_not_of_class_to_predict_labels = np.zeros(len(all_elements_not_of_class_to_predict[0,0,:]), dtype=np.int8)

  raw_title = title_of_run + '_raw'
  if '_raw' in modifier_for_run:
    return [np.concatenate((all_elements_of_class_to_predict[5]-all_elements_of_class_to_predict[0], all_elements_not_of_class_to_predict[5]-all_elements_not_of_class_to_predict[0]), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), raw_title]

  ### 02_ToDo: Modify all the 22 modalities through: [leave, CSP, CCAr, bCSTP]
  # Compute CSP
  csp_title = title_of_run + '_CSP'
  if '_CSP' in modifier_for_run:
    csp_filters, csp_eigenvals = meet.spatfilt.CSP(all_elements_of_class_to_predict[:32,:,:].mean(2), all_elements_not_of_class_to_predict[:32,:,:].mean(2))
    all_elements_of_class_to_predict_CSP_0 = np.tensordot(csp_filters[0].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_not_of_class_to_predict_CSP_0 = np.tensordot(csp_filters[0].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_of_class_to_predict_CSP_1 = np.tensordot(csp_filters[1].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_not_of_class_to_predict_CSP_1 = np.tensordot(csp_filters[1].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_of_class_to_predict_CSP_2 = np.tensordot(csp_filters[2].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_not_of_class_to_predict_CSP_2 = np.tensordot(csp_filters[2].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))
    return [np.concatenate((all_elements_of_class_to_predict_CSP_0, all_elements_not_of_class_to_predict_CSP_0), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), csp_title]

  # Compute CCAr
  ccar_title = title_of_run + '_CCAr'
  if '_CCAr' in modifier_for_run:
    a_ccar, b_ccar, s_ccar = meet.spatfilt.CCAvReg(all_elements_of_class_to_predict[:32,:,:])
    ccar_filt_all_elements_of_class_to_predict_0 = np.tensordot(a_ccar[:,0], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_not_of_class_to_predict_0 = np.tensordot(a_ccar[:,0], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_of_class_to_predict_1 = np.tensordot(a_ccar[:,1], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_not_of_class_to_predict_1 = np.tensordot(a_ccar[:,1], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_of_class_to_predict_2 = np.tensordot(a_ccar[:,2], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_not_of_class_to_predict_2 = np.tensordot(a_ccar[:,2], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))
    return [np.concatenate((ccar_filt_all_elements_of_class_to_predict_0, ccar_filt_all_elements_not_of_class_to_predict_0), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), ccar_title]

  # Compute bCSTP
  # s_bcstp_eigenvals, t_bcstp_eigenvals, W_bcstp, V_bcstp = bCSTP(all_elements_of_class_to_predict[:32,:,:], all_elements_not_of_class_to_predict[:32,:,:], num_iter=15, t_keep=3, s_keep=3)
  # left out as it would also need intrplt.data.... the scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_kx_data_combined_all_elements_of_class_to_predict[-1][:,0], intrplt_kx_data_combined[:8]), V_out_epoched_intrplt_kx_data_combined_all_elements_of_class_to_predict[-1][:,0][::-1], axis=-1)
  
  ### 03_ToDo: Modify all the 88 modalities through: [hil Y/N]
  hil_csp_title = title_of_run + '_CSP_hil'
  if '_CSP_hil' in modifier_for_run:
    csp_filters, csp_eigenvals = meet.spatfilt.CSP(all_elements_of_class_to_predict[:32,:,:].mean(2), all_elements_not_of_class_to_predict[:32,:,:].mean(2))
    all_elements_of_class_to_predict_CSP_0 = np.tensordot(csp_filters[0].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_not_of_class_to_predict_CSP_0 = np.tensordot(csp_filters[0].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_of_class_to_predict_CSP_1 = np.tensordot(csp_filters[1].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_not_of_class_to_predict_CSP_1 = np.tensordot(csp_filters[1].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_of_class_to_predict_CSP_2 = np.tensordot(csp_filters[2].T, all_elements_of_class_to_predict[:32,:,:], axes=(0 ,0))
    all_elements_not_of_class_to_predict_CSP_2 = np.tensordot(csp_filters[2].T, all_elements_not_of_class_to_predict[:32,:,:], axes=(0 ,0))

    hil_extracted_all_elements_of_class_to_predict_CSP_0 = calculate_hil_features(all_elements_of_class_to_predict_CSP_0)
    hil_extracted_all_elements_not_of_class_to_predict_CSP_0 = calculate_hil_features(all_elements_not_of_class_to_predict_CSP_0)
    hil_extracted_all_elements_of_class_to_predict_CSP_1 = calculate_hil_features(all_elements_of_class_to_predict_CSP_1)
    hil_extracted_all_elements_not_of_class_to_predict_CSP_1 = calculate_hil_features(all_elements_not_of_class_to_predict_CSP_1)
    hil_extracted_all_elements_of_class_to_predict_CSP_2 = calculate_hil_features(all_elements_of_class_to_predict_CSP_2)
    hil_extracted_all_elements_not_of_class_to_predict_CSP_2 = calculate_hil_features(all_elements_not_of_class_to_predict_CSP_2)
    hil_extracted_CSP_all_elements_of_class_to_predict = np.concatenate((hil_extracted_all_elements_of_class_to_predict_CSP_0, hil_extracted_all_elements_of_class_to_predict_CSP_1, hil_extracted_all_elements_of_class_to_predict_CSP_2), axis=0)
    hil_extracted_CSP_all_elements_not_of_class_to_predict = np.concatenate((hil_extracted_all_elements_not_of_class_to_predict_CSP_0, hil_extracted_all_elements_not_of_class_to_predict_CSP_1, hil_extracted_all_elements_not_of_class_to_predict_CSP_2), axis=0)
    return [np.concatenate((hil_extracted_CSP_all_elements_of_class_to_predict, hil_extracted_CSP_all_elements_not_of_class_to_predict), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), hil_csp_title]

  hil_ccar_title = title_of_run + '_CCAR_hil'
  if '_CCAR_hil' in modifier_for_run:
    a_ccar, b_ccar, s_ccar = meet.spatfilt.CCAvReg(all_elements_of_class_to_predict[:32,:,:])
    ccar_filt_all_elements_of_class_to_predict_0 = np.tensordot(a_ccar[:,0], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_not_of_class_to_predict_0 = np.tensordot(a_ccar[:,0], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_of_class_to_predict_1 = np.tensordot(a_ccar[:,1], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_not_of_class_to_predict_1 = np.tensordot(a_ccar[:,1], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_of_class_to_predict_2 = np.tensordot(a_ccar[:,2], all_elements_of_class_to_predict[:32,:,:], axes=(0, 0))
    ccar_filt_all_elements_not_of_class_to_predict_2 = np.tensordot(a_ccar[:,2], all_elements_not_of_class_to_predict[:32,:,:], axes=(0, 0))

    hil_extracted_ccar_filt_all_elements_of_class_to_predict_0 = calculate_hil_features(ccar_filt_all_elements_of_class_to_predict_0)
    hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_0 = calculate_hil_features(ccar_filt_all_elements_not_of_class_to_predict_0)
    hil_extracted_ccar_filt_all_elements_of_class_to_predict_1 = calculate_hil_features(ccar_filt_all_elements_of_class_to_predict_1)
    hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_1 = calculate_hil_features(ccar_filt_all_elements_not_of_class_to_predict_1)
    hil_extracted_ccar_filt_all_elements_of_class_to_predict_2 = calculate_hil_features(ccar_filt_all_elements_of_class_to_predict_2)
    hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_2 = calculate_hil_features(ccar_filt_all_elements_not_of_class_to_predict_2)
    hil_extracted_ccar_all_elements_of_class_to_predict = np.concatenate((hil_extracted_ccar_filt_all_elements_of_class_to_predict_0, hil_extracted_ccar_filt_all_elements_of_class_to_predict_1, hil_extracted_ccar_filt_all_elements_of_class_to_predict_2), axis=0)
    hil_extracted_ccar_all_elements_not_of_class_to_predict = np.concatenate((hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_0, hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_1, hil_extracted_ccar_filt_all_elements_not_of_class_to_predict_2), axis=0)
    return [np.concatenate((hil_extracted_ccar_all_elements_of_class_to_predict, hil_extracted_ccar_all_elements_not_of_class_to_predict), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), hil_ccar_title]

  # return the datasets in epoch, channel, time_in_channel - fashion
  return [
    [np.concatenate((all_elements_of_class_to_predict[5]-all_elements_of_class_to_predict[0], all_elements_not_of_class_to_predict[5]-all_elements_not_of_class_to_predict[0]), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), raw_title],
#   [np.concatenate((all_elements_of_class_to_predict.reshape(-1, all_elements_of_class_to_predict.shape[-1]), all_elements_not_of_class_to_predict.reshape(-1, all_elements_not_of_class_to_predict.shape[-1])), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), title_of_run + '_all_channels_flattened'],// I have no idea why, but this singular error still occurs
    [np.concatenate((all_elements_of_class_to_predict_CSP_0, all_elements_not_of_class_to_predict_CSP_0), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), csp_title],
    [np.concatenate((ccar_filt_all_elements_of_class_to_predict_0, ccar_filt_all_elements_not_of_class_to_predict_0), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), ccar_title],
    [np.concatenate((hil_extracted_CSP_all_elements_of_class_to_predict, hil_extracted_CSP_all_elements_not_of_class_to_predict), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), hil_csp_title],
    [np.concatenate((hil_extracted_ccar_all_elements_of_class_to_predict, hil_extracted_ccar_all_elements_not_of_class_to_predict), axis=1), np.concatenate((all_elements_of_class_to_predict_labels, all_elements_not_of_class_to_predict_labels), axis=0), hil_ccar_title]
  ]


callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_binary_accuracy', min_delta=0, patience=25, verbose=0, mode='max',
    baseline=None, restore_best_weights=True
)


## ToDo: Document, had to adjust to 512 in first Dense layer
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
    keras.layers.Dense(512, activation='relu'), 
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

runs_to_do = [
  ['epoched_filt_under_100_kx.npy', 'CSP_hil', 1],
  ['epoched_filt_over_100_kx.npy', 'raw', 1],
  ['epoched_filt_over_100_kx.npy', 'CCAr', 1],
  ['epoched_filt_over_100_kx.npy', 'CCAR_hil', 1],
  ['epoched_data.npy', 'CSP', 2],
  ['epoched_data.npy', 'CSP_hil', 2],
  ['epoched_filt_under_50_kx.npy', 'raw', 2],
  ['epoched_filt_under_50_kx.npy', 'CCAr', 2],
  ['epoched_filt_under_50_kx.npy', 'CCAR_hil', 2],
  ['epoched_filt_under_100_kx.npy', 'CSP', 2],
  ['epoched_filt_under_100_kx.npy', 'CSP_hil', 2],
  ['epoched_filt_over_100_kx.npy', 'raw', 2],
  ['epoched_filt_over_100_kx.npy', 'CCAr', 2],
  ['epoched_filt_over_100_kx.npy', 'CCAR_hil', 2],
  ['epoched_data.npy', 'CSP', 3],
  ['epoched_data.npy', 'CSP_hil', 3],
  ['epoched_filt_under_50_kx.npy', 'raw', 3],
  ['epoched_filt_under_50_kx.npy', 'CCAr', 3],
  ['epoched_filt_under_50_kx.npy', 'CCAR_hil', 3],
  ['epoched_filt_under_100_kx.npy', 'CSP', 3],
  ['epoched_filt_under_100_kx.npy', 'CSP_hil', 3],
  ['epoched_filt_over_100_kx.npy', 'raw', 3],
  ['epoched_filt_over_100_kx.npy', 'CCAr', 3],
  ['epoched_filt_over_100_kx.npy', 'CCAR_hil', 3]
]

for run_tuple in runs_to_do:
  modality_idx = run_tuple[0]
  modifier_for_run = run_tuple[1]
  class_id = run_tuple[2]

  workload = load_data_as_still_sorted_all_elements_of_class_to_predict_noise_data_then_labels_from_one_subject(modality_idx, modality_idx.split('.')[0] + '_w_leave_out_%s', id_to_leave_out, class_id, modifier_for_run)
  test_workload = load_test_data(modality_idx, modality_idx.split('.')[0] + '_w_leave_out_%s', id_to_leave_out, class_id, modifier_for_run)

  ctr = 0
  data = workload[0]
  labels = workload[1]
  run_title = workload[2]
  test_data = test_workload[0]
  test_labels = test_workload[1]

  print('run_title: %s' % run_title)
  print(data.shape)

  ### Shuffle and split data // .T is required to switch back to shape of (trial x feature)
  shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat)
  print(shuffled_data.shape)
  X_train, X_eval, y_train, y_eval = train_test_split(shuffled_data, shuffled_labels, test_size=0.25, random_state=rand_stat)
  print(X_train.shape)
  
  X_train = np.expand_dims(X_train, axis=-1)
  X_test = np.expand_dims(np.swapaxes(test_data, 0, 1), axis=-1)
  X_eval = np.expand_dims(X_eval, axis=-1)

  y_train = y_train.reshape((-1,1))
  y_test = test_labels.reshape((-1,1))
  y_eval = y_eval.reshape((-1,1))

  # some model definition and training
  model = create_alex_net(X_train.shape[1])
  history = model.fit(x=X_train, y=y_train, epochs=300, batch_size=32, validation_data=[X_eval, y_eval], callbacks=[callback]) 
  # model.save('/media/christoph/Volume/Masterthesis/DEAP_TSC/tsc_runs/1D/LOO/alex_net/models/alex_net_on_%s_predicting_class_%d' % (run_title.replace('/', '-'), class_id))
  np.save('/media/christoph/Volume/Masterthesis/DEAP_TSC/tsc_runs/1D/LOO/alex_net/histories/alex_net_on_%s_predicting_class_%d' % (run_title.replace('/', '-'), class_id), history.history)
  predictions = np.abs(np.rint(model.predict(x=X_test)))
  confusion_matrix_for_model = confusion_matrix(y_test, predictions)
  tn, fp, fn, tp = confusion_matrix_for_model.ravel()
  np.save('/media/christoph/Volume/Masterthesis/DEAP_TSC/tsc_runs/1D/LOO/alex_net/metrics/alex_net_on_%s_predicting_class_%d_confusion_matrix' % (run_title.replace('/', '-'), class_id), confusion_matrix_for_model)
  metrics = metrics_for_conf_mat(tn, fp, fn, tp)
  np.save('/media/christoph/Volume/Masterthesis/DEAP_TSC/tsc_runs/1D/LOO/alex_net/metrics/alex_net_on_%s_predicting_class_%d_metrics' % (run_title.replace('/', '-'), class_id), metrics)

  ctr += 1
  print('Change in counter-value from [%d] to [%d]' % (ctr-1, ctr))