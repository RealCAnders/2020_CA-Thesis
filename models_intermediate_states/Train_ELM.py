#!/usr/bin/python

import sys
import numpy as np
from numpy import load

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import meet

args = sys.argv

print('Number of arguments: %d arguments.' % len(args))
print('Argument List:', str(args))

idx = int(args[1])
rand_stat = 42

hfSEP_win = [50, 450]
noise_win = [-500, -100]

# ToDo: Load prepped data as epchd data w. outlier rejected and without
## ==> Gedanken machen zu "advanced prepped datasets": Wollen wir für dieselben CCAr jeweils auch noch mal durchlaufen lassen?
##													==> Oder wollen wir quasi je untersch. Filtermodalität die einzelnen Prep-Schritte (CSP, SSD, CCAr, bCSTP) durchführen und von denselben die hilbert-transform und darauffolgend die feature-extraction machen?
	# ToDo: Check if this loading and epoching here will be done correctly inside of the script
	# load triggers_identified_after_rejection = load('/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/triggers_identified_after_rejection.npy' % (idx + 1))
#[meet.epochEEG(load('/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_under_100_kx_out_rej.npy' % (idx + 1)), triggers_identified_after_rejection, hfSEP_win), meet.epochEEG(load('/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_under_100_kx_out_rej.npy' % (idx + 1)), triggers_identified_after_rejection, noise_win)]
#[meet.epochEEG(load('/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_over_100_kx_out_rej.npy' % (idx + 1)), triggers_identified_after_rejection, hfSEP_win), meet.epochEEG(load('/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_over_100_kx_out_rej.npy' % (idx + 1)), triggers_identified_after_rejection, noise_win)]
#[meet.epochEEG(load('/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_over_400_kx_kx_out_rej.npy' % (idx + 1)), triggers_identified_after_rejection, hfSEP_win), meet.epochEEG(load('/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_over_400_kx_kx_out_rej.npy' % (idx + 1)), triggers_identified_after_rejection, noise_win)]
#[meet.epochEEG(load('/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_500_900_kx_out_rej.npy' % (idx + 1)), triggers_identified_after_rejection, hfSEP_win), meet.epochEEG(load('/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_500_900_kx_out_rej.npy' % (idx + 1)), triggers_identified_after_rejection, noise_win)]

# ToDo: Adjust so that code runs through the different modalities
# ToDo: Include code for feature-extraction and use the outlier-rejected as well as non-outlier-rejected datasets too
# hfSEP, noise, title_of_data
outlier_rejected_datasets_per_subject = [
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_kx_data_combined'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_noise_out_rej.npy', 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_kx_data_combined'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_noise_out_rej.npy', 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_noise_out_rej.npy', 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_noise_out_rej.npy', 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_noise_out_rej.npy', 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_kx_data_combined_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_kx_data_combined_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_kx_data_combined'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_under_100_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_100_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_400_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_kx_data_combined_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_kx_data_combined_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_kx_data_combined'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_under_100_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_100_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_400_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_500_900_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx'],
	['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx']
]

def load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_path, noise_path, title, identifier):
	hfSEP = load(hfsep_path % (identifier + 1))
	noise = load(noise_path % (identifier + 1))
	title_of_run = title % (identifier + 1)

	hfsep_labels = np.ones(len(hfSEP[0,0,:]), dtype=np.int8)
	noise_labels = np.zeros(len(noise[0,0,:]), dtype=np.int8)

	# return them in epoch, channel, time_in_channel - fashion
	return [np.swapaxes(np.swapaxes(np.concatenate((hfSEP[:8], noise[:8]), axis=2), 0, 1), 0, 2), np.concatenate((hfsep_labels, noise_labels), axis=0), title_of_run]

### Loop over the different data-modalities and compute the confusion-matrix for each
## load the data, start with k0010, then go the range until k009 create a reproducible datasplit and shuffle the samples then

data, labels, run_title = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(idx)
print(data.shape)
shuffled_data, shuffled_labels = shuffle(data, labels, random_state=rand_stat)
print(shuffled_data.shape)
X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat)
print(X_train.shape)

## Reshape so that we have the concatenated datapoints of each modality/channel appended together. five-channels á 400 datapoints would yield (trial x 2000)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
print(X_train.shape)

## Some model definition
elm = meet.elm.ClassELM()

## Train and save the model
elm.cv(X_train, y_train)
if elm.istrained:
	print('Succesfully trained the ELM. Now work on gaining insights and getting performance metrics.')
	print('Performing classification on the test-data now. Result:')
	print(meet.elm.get_conf_matrix(y_test, elm.classify(X_test)))

## Next up, thereafter, perform the hilbert-transformation of all the individual datasets and compute w. 12-chan data.
"""
Extract features as in Paper by Gunnar, slightly modified w.r.t. check on which data will be used:

# ['/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx']
### Load data of k003, to check on performance and compare w. results pusblished
data, labels, run_title = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject('/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/intermediate_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx', 2)
print('run_title: %s' % run_title)
print(data.shape)

### Load data like in Train_ELM
data_shape_for_ccar = np.swapaxes(np.swapaxes(data, 0, 1), 1, 2)

### Compute CCAr
a_ccar, b_ccar, s_ccar = meet.spatfilt.CCAvReg(data_shape_for_ccar[:,:,labels==1])

### Repeat this process for the three best (idx == 0, 1, 2) CCAr-Filters 
ccar_filt_dat_k0010_0 = np.tensordot(a_ccar[:,0], data_shape_for_ccar, axes=(0, 0))
hil_ccar_filt_dat_k0010_0 = scipy.signal.hilbert(ccar_filt_dat_k0010_0, axis=0)
real_hil_ccar_filt_dat_k0010_0 = np.real(hil_ccar_filt_dat_k0010_0)
imag_hil_ccar_filt_dat_k0010_0 = np.imag(hil_ccar_filt_dat_k0010_0)
abs_hil_ccar_filt_dat_k0010_0 = np.abs(hil_ccar_filt_dat_k0010_0)
angle_hil_ccar_filt_dat_k0010_0 = np.angle(hil_ccar_filt_dat_k0010_0)

ccar_filt_dat_k0010_1 = np.tensordot(a_ccar[:,1], data_shape_for_ccar, axes=(0, 0))
hil_ccar_filt_dat_k0010_1 = scipy.signal.hilbert(ccar_filt_dat_k0010_1, axis=0)
real_hil_ccar_filt_dat_k0010_1 = np.real(hil_ccar_filt_dat_k0010_1)
imag_hil_ccar_filt_dat_k0010_1 = np.imag(hil_ccar_filt_dat_k0010_1)
abs_hil_ccar_filt_dat_k0010_1 = np.abs(hil_ccar_filt_dat_k0010_1)
angle_hil_ccar_filt_dat_k0010_1 = np.angle(hil_ccar_filt_dat_k0010_1)

ccar_filt_dat_k0010_2 = np.tensordot(a_ccar[:,0], data_shape_for_ccar, axes=(0, 0))
hil_ccar_filt_dat_k0010_2 = scipy.signal.hilbert(ccar_filt_dat_k0010_2, axis=0)
real_hil_ccar_filt_dat_k0010_2 = np.real(hil_ccar_filt_dat_k0010_2)
imag_hil_ccar_filt_dat_k0010_2 = np.imag(hil_ccar_filt_dat_k0010_2)
abs_hil_ccar_filt_dat_k0010_2 = np.abs(hil_ccar_filt_dat_k0010_2)
angle_hil_ccar_filt_dat_k0010_2 = np.angle(hil_ccar_filt_dat_k0010_2)

### In Pseudo-Code continued:
# Add all these three times four transformed channels together as new feature vector of length:
extracted_features = np.concatenate((real_hil_ccar_filt_dat_k0010_0, imag_hil_ccar_filt_dat_k0010_0, abs_hil_ccar_filt_dat_k0010_0, angle_hil_ccar_filt_dat_k0010_0, real_hil_ccar_filt_dat_k0010_1, imag_hil_ccar_filt_dat_k0010_1, abs_hil_ccar_filt_dat_k0010_1, angle_hil_ccar_filt_dat_k0010_1, real_hil_ccar_filt_dat_k0010_2, imag_hil_ccar_filt_dat_k0010_2, abs_hil_ccar_filt_dat_k0010_2, angle_hil_ccar_filt_dat_k0010_2), axis=0)

### Shuffle and split data // .T is required to switch back to shape of (trial x feature)
shuffled_data_hil, shuffled_labels_hil = shuffle(extracted_features.T, labels, random_state=rand_stat)
print(shuffled_data_hil.shape)
X_train_hil, X_test_hil, y_train_hil, y_test_hil = train_test_split(shuffled_data_hil, shuffled_labels_hil, test_size=0.33, random_state=rand_stat)
print(X_train_hil.shape)

## Some model definition
elm = meet.elm.ClassELM()

### Train ELM on the extracted features using cv for hyperparameter and then classify test-samples
elm.cv(X_train_hil, y_train_hil)
if elm.istrained:
	print('Succesfully trained the ELM. Now work on gaining insights and getting performance metrics.')
	print('Performing classification on the test-data now. Result:')
	print(meet.elm.get_conf_matrix(y_test_hil, elm.classify(X_test_hil)))
"""