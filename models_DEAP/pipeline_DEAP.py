import sys
import meet  
import numpy as np  
import matplotlib.pyplot as plt  
import scipy
from scipy import signal  
from scipy.fft import fftshift  
from scipy.ndimage import convolve1d, convolve  
from numpy import save
from meet.spatfilt import CSP
import mne
import pickle

### ### ### ### ###
### General Idea description on DEAP dataset
### https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html#orig
### ### ### ### ###
# If we identified 40 blocks of status code '1' in the raw-data
# then we can simply assume the 3rd element to be start of status recording
# and the 1st to be start of video; for CCAr etc. we use noise in 1st elem
# each block used will be 10s, so 512 * 10, as sampling_rate = 512

# For CCAr we need the trials to compute the average for, so we use all the ones
# where the class-score for class C is highest

### ### ### ### ###
### Definition: utility-function / global vars
### ### ### ### ###
s_rate = 512


def identify_triggers(trigger_signal, estimated_trigger_distance, indicator_value):  
  
    # 1st version: define the timestamp when the signal is at zero again as "start of trigger"  
    triggers = [0]  
    ttl_found = False  
    ttl_samples_ctr = 0  
      
    for idx, data_point in enumerate(trigger_signal):  
        if triggers[-1] + int(0.9 * estimated_trigger_distance) <= idx and trigger_signal[idx] == indicator_value:  
            ttl_found = True  
            ttl_samples_ctr = ttl_samples_ctr + 1  
        else:  
            ttl_found = False  
        if ttl_samples_ctr > 0 and not ttl_found:  
            triggers.append(idx)
            ttl_samples_ctr = 0  
          
    return triggers[1:] 


def give_tuples_of_length(start, end, length): 
    elements = [] 
    for el in range(start, end, 512*5)[:5]: 
        elements.append(el) 
    return elements 

# ToDo: data instead of data
# ToDo: Indices + 1 instead of + 2
def get_indices_for_rating_screen(subject_id, triggers):
  intermediate_triggers = [[triggers[x], triggers[x+3]] for x in range(0, len(triggers), 4)]
  intermediate_elements = [elem[0] if len(elem[0]) == 5 else None for elem in [[give_tuples_of_length(start, end, 512*5)] for start, end in intermediate_triggers]]

  indices_of_elements_not_none = []

  for i in intermediate_elements:
    if i is not None:
      indices_of_elements_not_none.append(True)
    else: 
      indices_of_elements_not_none.append(False)

  intermediate_elements = np.asarray(intermediate_elements)
  triggers_start_indizes_wanted = intermediate_elements[intermediate_elements != np.array(None)]

  ### Load labels recomputed using threshold of 5.0 --> 0 or 1 // into (subject, trial, class) - shape
  labels_per_participant = np.loadtxt(open("/media/christoph/Volume/Masterthesis/DEAP_TSC/DEAP_labels_per_participant.csv", "rb"), delimiter=",")
  labels_per_participant = labels_per_participant.reshape(32, 40, 4)
  indices_of_elements_not_none = np.asarray(indices_of_elements_not_none)

  return [np.asarray(labels_per_participant[subject_id, indices_of_elements_not_none == np.array(True),:]), np.concatenate(triggers_start_indizes_wanted, axis=0)]


def get_indices_for_recording_screen(triggers):
  return [triggers[x] for x in range(0, len(triggers), 4)]


def normalize_min_max(to_be_normalized): 
    return (to_be_normalized - np.min(to_be_normalized)) / (np.max(to_be_normalized) - np.min(to_be_normalized)) 


def normalize_z(to_be_normalized): 
    return (to_be_normalized - np.mean(to_be_normalized)) / (np.std(to_be_normalized)) 


# ToDo: Adjust so that s0x is not anymore saved as s0x!!
### ### ### ### ###
### Data loading
### ### ### ### ###
idx = int(sys.argv[1])
print('Running final data prepping for s0%d ...' % idx)

if idx < 10:
  data_path = '/home/christoph/Desktop/Data_Thesis_Analyze/DEAP/data_original/s0%d.bdf'
else:
  data_path = '/home/christoph/Desktop/Data_Thesis_Analyze/DEAP/data_original/s%d.bdf'

win_end = s_rate * 5
data_win = [0, win_end]

# load data from path, only 32-EEG-channels and append status channel
s0x_raw = mne.io.read_raw_bdf(input_fname=(data_path % idx), exclude=[33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], preload=True)
data, _ = s0x_raw[:]
data_s0x = np.concatenate((data[:32], np.expand_dims(data[-1], axis=0)), axis=0)

# get triggers but start after first calibration phase, so after 64000
# 1-22 need to have 10, 1;; 23 - 32 need to have 10, 1638145
triggers_s0x = np.asarray(identify_triggers(data_s0x[32,:], 10, 1))[-160:]
# triggers_s0x = np.asarray(identify_triggers(data_s0x[32,:], 10, 1638145))[-160:]

save('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/data_s0x' % idx, data_s0x)
save('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/triggers_s0x' % idx, triggers_s0x)
print('Stored basic versions of data for S%d' % idx)

### ### ### ### ###
### Preprocess without outlier-rejection
### ### ### ### ###
intrplt_filt_under_50_kx = meet.iir.butterworth(data_s0x[:32], fp=50, fs=55, s_rate=512)
intrplt_filt_under_50_kx = np.append(intrplt_filt_under_50_kx, np.expand_dims(data_s0x[-1], axis=0), axis=0)
save('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/intrplt_filt_under_50_kx' % idx, intrplt_filt_under_50_kx)

intrplt_filt_under_100_kx = meet.iir.butterworth(data_s0x[:32], fp=100, fs=110, s_rate=512)
intrplt_filt_under_100_kx = np.append(intrplt_filt_under_100_kx, np.expand_dims(data_s0x[-1], axis=0), axis=0)
save('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/intrplt_filt_under_100_kx' % idx, intrplt_filt_under_100_kx)

intrplt_filt_over_100_kx = meet.iir.butterworth(data_s0x[:32], fp=100, fs=90, s_rate=512)
intrplt_filt_over_100_kx = np.append(intrplt_filt_over_100_kx, np.expand_dims(data_s0x[-1], axis=0), axis=0)
save('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/intrplt_filt_over_100_kx' % idx, intrplt_filt_over_100_kx)

intrplt_filt_over_200_kx = meet.iir.butterworth(data_s0x[:32], fp=200, fs=180, s_rate=512)
intrplt_filt_over_200_kx = np.append(intrplt_filt_over_200_kx, np.expand_dims(data_s0x[-1], axis=0), axis=0)
save('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/intrplt_filt_over_200_kx' % idx, intrplt_filt_over_200_kx)

labels_and_indices = get_indices_for_rating_screen(idx, triggers_s0x)
labels = labels_and_indices[0]
indices_for_ratings = labels_and_indices[1]
print('Shape of labels for s%s' % idx)
print(labels.shape)
print('Shape of elements for s%s' % idx)
print(indices_for_ratings.shape)

save('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/labels' % idx, labels)

epoched_data_s0x = meet.epochEEG(data_s0x[:32], indices_for_ratings, data_win)

print(epoched_data_s0x.shape)     ## (32, 5120, 40) // (channel, time-in-trial, trial)
save('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/epoched_data' % idx, epoched_data_s0x)

epoched_filt_under_50_kx = meet.epochEEG(intrplt_filt_under_50_kx, indices_for_ratings, data_win)
save('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/epoched_filt_under_50_kx' % idx, epoched_filt_under_50_kx)

epoched_filt_under_100_kx = meet.epochEEG(intrplt_filt_under_100_kx, indices_for_ratings, data_win)
save('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/epoched_filt_under_100_kx' % idx, epoched_filt_under_100_kx)

epoched_filt_over_100_kx = meet.epochEEG(intrplt_filt_over_100_kx, indices_for_ratings, data_win)
save('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/epoched_filt_over_100_kx' % idx, epoched_filt_over_100_kx)

epoched_filt_over_200_kx = meet.epochEEG(intrplt_filt_over_200_kx, indices_for_ratings, data_win)
save('/media/christoph/Volume/Masterthesis/DEAP_TSC/data/s%d/epoched_filt_over_200_kx' % idx, epoched_filt_over_200_kx)

print('For k00%d basic preprocessing without outlier-rejection made' % idx)

######
###### Continue tomorrow with these approaches, 
###### taking the data preprocessed, loading it, 
###### let it be worked through in four 
###### binary-classification tasks, but using MCC as unbalanced!
######
#-#
#-#In [34]: noise_epoched_data_s0x = epoched_data_s0x_hfsep[:,:,labels_per_participant[0, :, 0] == 1]                                                                          
#-#
#-#In [35]: noise_epoched_data_s0x.shape                                                                                                                                       
#-#Out[35]: (32, 5120, 19)
#-#
#-#In [36]: dominance_epoched_data_s0x = epoched_data_s0x_hfsep[:,:,labels_per_participant[0, :, 2] == 1]                                                                      
#-#
#-#In [37]: dominance_epoched_data_s0x.shape                                                                                                                                   
#-#Out[37]: (32, 5120, 26)
#-#
######
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ### ### ### ### ###
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ### Advanced Preprocessing on data without outlier-rejection
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ### ### ### ### ###
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# # CSP is the signal decomposition using two different signal modalities, due to different points in time, but same preprocessing
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# print('Shape of epoched_data_s0x:')
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# print(epoched_data_s0x.shape)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_epoched_data_s0x_filters, csp_epoched_data_s0x_eigenvals = meet.spatfilt.CSP(epoched_data_s0x[:8].reshape(8, -1), epoched_data_s0x_noise[:8].reshape(8, -1))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_epoched_data_s0x_filters, csp_epoched_data_s0x_eigenvals = meet.spatfilt.CSP(epoched_data_s0x[:8].reshape(8, -1), epoched_data_s0x_noise[:8].reshape(8, -1))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_epoched_filt_under_100_kx_filters, csp_epoched_filt_under_100_kx_eigenvals = meet.spatfilt.CSP(epoched_filt_under_100_kx[:8].reshape(8, -1), epoched_filt_under_100_kx_noise[:8].reshape(8, -1))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_epoched_filt_over_100_kx_filters, csp_epoched_filt_over_100_kx_eigenvals = meet.spatfilt.CSP(epoched_filt_over_100_kx[:8].reshape(8, -1), epoched_filt_over_100_kx_noise[:8].reshape(8, -1))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_epoched_filt_under_50_kx_filters, csp_epoched_filt_under_50_kx_eigenvals = meet.spatfilt.CSP(epoched_filt_under_50_kx[:8].reshape(8, -1), epoched_filt_under_50_kx_noise[:8].reshape(8, -1))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_epoched_filt_500_900_kx_filters, csp_epoched_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_filt_500_900_kx[:8].reshape(8, -1), epoched_filt_500_900_kx_noise[:8].reshape(8, -1))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_data_s0x = np.tensordot(csp_epoched_data_s0x_filters[:,0].T, data_s0x[:8], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_data_s0x = np.tensordot(csp_epoched_data_s0x_filters[:,0].T, intrplt_data_s0x[:8], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_filt_under_100_kx = np.tensordot(csp_epoched_filt_under_100_kx_filters[:,0].T, intrplt_filt_under_100_kx[:8], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_filt_over_100_kx = np.tensordot(csp_epoched_filt_over_100_kx_filters[:,0].T, intrplt_filt_over_100_kx[:8], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_filt_under_50_kx = np.tensordot(csp_epoched_filt_under_50_kx_filters[:,0].T, intrplt_filt_under_50_kx[:8], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_filt_500_900_kx = np.tensordot(csp_epoched_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_data_s0x = meet.epochEEG(csp_filt_epoched_data_s0x, triggers_for_kx_combined, data_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_data_s0x = meet.epochEEG(csp_filt_epoched_data_s0x, triggers_for_kx_combined, data_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_filt_under_100_kx = meet.epochEEG(csp_filt_epoched_filt_under_100_kx, triggers_for_kx_combined, data_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_filt_over_100_kx = meet.epochEEG(csp_filt_epoched_filt_over_100_kx, triggers_for_kx_combined, data_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_filt_under_50_kx = meet.epochEEG(csp_filt_epoched_filt_under_50_kx, triggers_for_kx_combined, data_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_filt_500_900_kx = meet.epochEEG(csp_filt_epoched_filt_500_900_kx, triggers_for_kx_combined, data_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_data_s0x_noise = meet.epochEEG(csp_filt_epoched_data_s0x, get_indices_for_noise(triggers_for_kx_combined), noise_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_data_s0x_noise = meet.epochEEG(csp_filt_epoched_data_s0x, get_indices_for_noise(triggers_for_kx_combined), noise_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_filt_under_100_kx_noise = meet.epochEEG(csp_filt_epoched_filt_under_100_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_filt_over_100_kx_noise = meet.epochEEG(csp_filt_epoched_filt_over_100_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_filt_under_50_kx_noise = meet.epochEEG(csp_filt_epoched_filt_under_50_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# csp_filt_epoched_filt_500_900_kx_noise = meet.epochEEG(csp_filt_epoched_filt_500_900_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_data_s0x' % idx, csp_filt_epoched_data_s0x)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_data_s0x' % idx, csp_filt_epoched_data_s0x)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_filt_under_100_kx' % idx, csp_filt_epoched_filt_under_100_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_filt_over_100_kx' % idx, csp_filt_epoched_filt_over_100_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_filt_under_50_kx' % idx, csp_filt_epoched_filt_under_50_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_filt_500_900_kx' % idx, csp_filt_epoched_filt_500_900_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_data_s0x_noise' % idx, csp_filt_epoched_data_s0x_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_data_s0x_noise' % idx, csp_filt_epoched_data_s0x_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_filt_under_100_kx_noise' % idx, csp_filt_epoched_filt_under_100_kx_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_filt_over_100_kx_noise' % idx, csp_filt_epoched_filt_over_100_kx_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_filt_under_50_kx_noise' % idx, csp_filt_epoched_filt_under_50_kx_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_filt_500_900_kx_noise' % idx, csp_filt_epoched_filt_500_900_kx_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# print('For k00%d CSP without outlier-rejection made' % idx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# # SSD is the signal decomposition using two different frequency-filtered datasets as different conditions
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# print('Skipping SSD for now')
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######  ssd_intrplt_filt_under_100_kx_filters, ssd_intrplt_filt_under_100_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_under_100_kx[:8], intrplt_data_s0x[:8])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######  ssd_intrplt_filt_over_100_kx_filters, ssd_intrplt_filt_over_100_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_over_100_kx[:8], intrplt_data_s0x[:8])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######  ssd_intrplt_filt_under_50_kx_filters, ssd_intrplt_filt_under_50_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_under_50_kx[:8], intrplt_data_s0x[:8])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######  ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx[:8], intrplt_data_s0x[:8])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######  ssd_filt_intrplt_filt_under_100_kx = ssd_intrplt_filt_under_100_kx_filters[:,0].T.dot(intrplt_filt_under_100_kx[:8])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######  ssd_filt_intrplt_filt_over_100_kx = ssd_intrplt_filt_over_100_kx_filters.T.dot(intrplt_filt_over_100_kx[:8])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######  ssd_filt_intrplt_filt_under_50_kx_kx = ssd_intrplt_filt_under_50_kx_filters.T.dot(intrplt_filt_under_50_kx[:8])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######  ssd_filt_intrplt_filt_500_900_kx = ssd_intrplt_filt_500_900_kx_filters.T.dot(intrplt_filt_500_900_kx[:8])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_under_100_kx' % idx, ssd_filt_intrplt_filt_under_100_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_100_kx' % idx, ssd_filt_intrplt_filt_over_100_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_under_50_kx_kx' % idx, ssd_filt_intrplt_filt_under_50_kx_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ######  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_500_900_kx' % idx, ssd_filt_intrplt_filt_500_900_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# print('For k00%d SSD without outlier-rejection made' % idx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# # CCAr is the technique that tries to derive filters that 'modify' the single-trial to be more similar to the single-trial averages
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# a_epoched_data_s0x, b_epoched_data_s0x, s_epoched_data_s0x = meet.spatfilt.CCAvReg(epoched_data_s0x[:8,:,:])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# a_epoched_data_s0x, b_epoched_data_s0x, s_epoched_data_s0x = meet.spatfilt.CCAvReg(epoched_data_s0x[:8,:,:])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# a_epoched_filt_under_100_kx, b_epoched_filt_under_100_kx, s_epoched_filt_under_100_kx = meet.spatfilt.CCAvReg(epoched_filt_under_100_kx[:8,:,:])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# a_epoched_filt_over_100_kx, b_epoched_filt_over_100_kx, s_epoched_filt_over_100_kx = meet.spatfilt.CCAvReg(epoched_filt_over_100_kx[:8,:,:])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# a_epoched_filt_under_50_kx, b_epoched_filt_under_50_kx, s_epoched_filt_under_50_kx = meet.spatfilt.CCAvReg(epoched_filt_under_50_kx[:8,:,:])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# a_epoched_filt_500_900_kx, b_epoched_filt_500_900_kx, s_epoched_filt_500_900_kx = meet.spatfilt.CCAvReg(epoched_filt_500_900_kx[:8,:,:])
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ccar_filt_epoched_data_s0x = np.tensordot(a_epoched_data_s0x[:,0], epoched_data_s0x[:8,:,:], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ccar_filt_epoched_data_s0x_noise = np.tensordot(a_epoched_data_s0x[:,0], epoched_data_s0x_noise[:8,:,:], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ccar_filt_epoched_data_s0x = np.tensordot(a_epoched_data_s0x[:,0], epoched_data_s0x[:8,:,:], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ccar_filt_epoched_data_s0x_noise = np.tensordot(a_epoched_data_s0x[:,0], epoched_data_s0x_noise[:8,:,:], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ccar_filt_epoched_filt_under_100_kx = np.tensordot(a_epoched_filt_under_100_kx[:,0], epoched_filt_under_100_kx[:8,:,:], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ccar_filt_epoched_filt_under_100_kx_noise = np.tensordot(a_epoched_filt_under_100_kx[:,0], epoched_filt_under_100_kx_noise[:8,:,:], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ccar_filt_epoched_filt_over_100_kx = np.tensordot(a_epoched_filt_over_100_kx[:,0], epoched_filt_over_100_kx[:8,:,:], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ccar_filt_epoched_filt_over_100_kx_noise = np.tensordot(a_epoched_filt_over_100_kx[:,0], epoched_filt_over_100_kx_noise[:8,:,:], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ccar_filt_epoched_filt_under_50_kx = np.tensordot(a_epoched_filt_under_50_kx[:,0], epoched_filt_under_50_kx[:8,:,:], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ccar_filt_epoched_filt_under_50_kx_noise = np.tensordot(a_epoched_filt_under_50_kx[:,0], epoched_filt_under_50_kx_noise[:8,:,:], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ccar_filt_epoched_filt_500_900_kx = np.tensordot(a_epoched_filt_500_900_kx[:,0], epoched_filt_500_900_kx[:8,:,:], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# ccar_filt_epoched_filt_500_900_kx_noise = np.tensordot(a_epoched_filt_500_900_kx[:,0], epoched_filt_500_900_kx_noise[:8,:,:], axes=(0, 0))
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_data_s0x' % idx, ccar_filt_epoched_data_s0x)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_data_s0x_noise' % idx, ccar_filt_epoched_data_s0x_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_data_s0x' % idx, ccar_filt_epoched_data_s0x)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_data_s0x_noise' % idx, ccar_filt_epoched_data_s0x_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_filt_under_100_kx' % idx, ccar_filt_epoched_filt_under_100_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_filt_under_100_kx_noise' % idx, ccar_filt_epoched_filt_under_100_kx_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_filt_over_100_kx' % idx, ccar_filt_epoched_filt_over_100_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_filt_over_100_kx_noise' % idx, ccar_filt_epoched_filt_over_100_kx_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_filt_under_50_kx' % idx, ccar_filt_epoched_filt_under_50_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_filt_under_50_kx_noise' % idx, ccar_filt_epoched_filt_under_50_kx_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_filt_500_900_kx' % idx, ccar_filt_epoched_filt_500_900_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_filt_500_900_kx_noise' % idx, ccar_filt_epoched_filt_500_900_kx_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# print('For k00%d CCAr without outlier-rejection made' % idx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# # bCSPT is the technique that tries to derive filters in the spatial and in the temporal domain, leading to the use of convolution
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# s_outepoched_data_s0x_eigenvals, t_outepoched_data_s0x_eigenvals, W_out_epoched_data_s0x, V_out_epoched_data_s0x = bCSTP(epoched_data_s0x[:8,:,:], epoched_data_s0x_noise[:8,:,:], num_iter=15, t_keep=2, s_keep=2)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# s_outepoched_filt_under_100_kx_eigenvals, t_outepoched_filt_under_100_kx_eigenvals, W_out_epoched_filt_under_100_kx, V_out_epoched_filt_under_100_kx = bCSTP(epoched_filt_under_100_kx[:8,:,:], epoched_filt_under_100_kx_noise[:8,:,:], num_iter=15, t_keep=2, s_keep=2)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# s_outepoched_filt_over_100_kx_eigenvals, t_outepoched_filt_over_100_kx_eigenvals, W_out_epoched_filt_over_100_kx, V_out_epoched_filt_over_100_kx = bCSTP(epoched_filt_over_100_kx[:8,:,:], epoched_filt_over_100_kx_noise[:8,:,:], num_iter=15, t_keep=2, s_keep=2)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# s_outepoched_filt_under_50_kx_eigenvals, t_outepoched_filt_under_50_kx_eigenvals, W_out_epoched_filt_under_50_kx, V_out_epoched_filt_under_50_kx = bCSTP(epoched_filt_under_50_kx[:8,:,:], epoched_filt_under_50_kx_noise[:8,:,:], num_iter=15, t_keep=2, s_keep=2)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# s_outepoched_filt_500_900_kx_eigenvals, t_outepoched_filt_500_900_kx_eigenvals, W_out_epoched_filt_500_900_kx, V_out_epoched_filt_500_900_kx = bCSTP(epoched_filt_500_900_kx[:8,:,:], epoched_filt_500_900_kx_noise[:8,:,:], num_iter=15, t_keep=2, s_keep=2)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_data_s0x = scipy.ndimage.convolve1d(np.dot(W_out_epoched_data_s0x[-1][:,0], intrplt_data_s0x[:8]), V_out_epoched_data_s0x[-1][:,0][::-1], axis=-1)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_filt_under_100_kx = scipy.ndimage.convolve1d(np.dot(W_out_epoched_filt_under_100_kx[-1][:,0], intrplt_filt_under_100_kx[:8]), V_out_epoched_filt_under_100_kx[-1][:,0][::-1], axis=-1)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_filt_over_100_kx = scipy.ndimage.convolve1d(np.dot(W_out_epoched_filt_over_100_kx[-1][:,0], intrplt_filt_over_100_kx[:8]), V_out_epoched_filt_over_100_kx[-1][:,0][::-1], axis=-1)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_filt_under_50_kx = scipy.ndimage.convolve1d(np.dot(W_out_epoched_filt_under_50_kx[-1][:,0], intrplt_filt_under_50_kx[:8]), V_out_epoched_filt_under_50_kx[-1][:,0][::-1], axis=-1)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_filt_500_900_kx = scipy.ndimage.convolve1d(np.dot(W_out_epoched_filt_500_900_kx[-1][:,0], intrplt_filt_500_900_kx[:8]), V_out_epoched_filt_500_900_kx[-1][:,0][::-1], axis=-1)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_data_s0x = meet.epochEEG(bcstp_spat_temp_filt_epoched_data_s0x, triggers_for_kx_combined, data_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_data_s0x_noise = meet.epochEEG(bcstp_spat_temp_filt_epoched_data_s0x, get_indices_for_noise(triggers_for_kx_combined), noise_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_filt_under_100_kx = meet.epochEEG(bcstp_spat_temp_filt_epoched_filt_under_100_kx, triggers_for_kx_combined, data_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_filt_under_100_kx_noise = meet.epochEEG(bcstp_spat_temp_filt_epoched_filt_under_100_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_filt_over_100_kx = meet.epochEEG(bcstp_spat_temp_filt_epoched_filt_over_100_kx, triggers_for_kx_combined, data_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_filt_over_100_kx_noise = meet.epochEEG(bcstp_spat_temp_filt_epoched_filt_over_100_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_filt_under_50_kx = meet.epochEEG(bcstp_spat_temp_filt_epoched_filt_under_50_kx, triggers_for_kx_combined, data_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_filt_under_50_kx_noise = meet.epochEEG(bcstp_spat_temp_filt_epoched_filt_under_50_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_filt_500_900_kx = meet.epochEEG(bcstp_spat_temp_filt_epoched_filt_500_900_kx, triggers_for_kx_combined, data_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# bcstp_spat_temp_filt_epoched_filt_500_900_kx_noise = meet.epochEEG(bcstp_spat_temp_filt_epoched_filt_500_900_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_data_s0x' % idx, bcstp_spat_temp_filt_epoched_data_s0x)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_data_s0x_noise' % idx, bcstp_spat_temp_filt_epoched_data_s0x_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_filt_under_100_kx' % idx, bcstp_spat_temp_filt_epoched_filt_under_100_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_filt_under_100_kx_noise' % idx, bcstp_spat_temp_filt_epoched_filt_under_100_kx_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_filt_over_100_kx' % idx, bcstp_spat_temp_filt_epoched_filt_over_100_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_filt_over_100_kx_noise' % idx, bcstp_spat_temp_filt_epoched_filt_over_100_kx_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_filt_under_50_kx' % idx, bcstp_spat_temp_filt_epoched_filt_under_50_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_filt_under_50_kx_noise' % idx, bcstp_spat_temp_filt_epoched_filt_under_50_kx_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_filt_500_900_kx' % idx, bcstp_spat_temp_filt_epoched_filt_500_900_kx)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_filt_500_900_kx_noise' % idx, bcstp_spat_temp_filt_epoched_filt_500_900_kx_noise)
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# 
#-#-# Skipped due to incorporating this in the TSC-Scripts #-#-# print('For k00%d bCSTP without outlier-rejection made' % idx)

### ### ### ### ###
### End of Data Preprocessing Script
### ### ### ### ###