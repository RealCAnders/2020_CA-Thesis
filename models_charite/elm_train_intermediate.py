import sys
import scipy
import pickle
import numpy as np

from numpy import load
from scipy.fft import fftshift  
from scipy.ndimage import convolve1d, convolve 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import meet

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

# ToDo: Adjust so that code runs through the different modalities
### 00_ToDo: Gather all the 11 base-modalities (leaving out for now the Z-Normalization)
base_modalities_per_subject = [
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise.npy', 'k00%d/epoched_kx_data_combined'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise.npy', 'k00%d/epoched_intrplt_kx_data_combined'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise.npy', 'k00%d/epoched_intrplt_filt_under_100_kx'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise.npy', 'k00%d/epoched_intrplt_filt_over_100_kx'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise.npy', 'k00%d/epoched_intrplt_filt_over_400_kx'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy', 'k00%d/epoched_intrplt_filt_500_900_kx'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_out_rej'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_out_rej'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_out_rej'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_out_rej'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_out_rej'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_noise_out_rej.npy', 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_kx_data_combined'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_noise_out_rej.npy', 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_noise_out_rej.npy', 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_noise_out_rej.npy', 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_noise_out_rej.npy', 'z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx'],
]

# 	ToDo: Incorporate also the Z-Normalized datasets
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_kx_data_combined_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_kx_data_combined_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_kx_data_combined'],
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_under_100_kx'],
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_100_kx'],
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_400_kx'],
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx'],
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_kx_data_combined_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_kx_data_combined_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_kx_data_combined'],
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_under_100_kx'],
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_100_kx'],
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_400_kx'],
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_500_900_kx'],
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx'],
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx'],
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx'],
#	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej.npy', 'advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_e#poched_intrplt_filt_500_900_kx']

def calculate_hil_features(transformed):
	hil_dat = scipy.signal.hilbert(transformed, axis=0)
	real_hil_dat = np.real(hil_dat)
	imag_hil_dat = np.imag(hil_dat)
	abs_hil_dat = np.abs(hil_dat)
	angle_hil_dat = np.angle(hil_dat)
	return np.concatenate((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat), axis=0)

def load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_path, noise_path, title, identifier):
	# load data in format of (channel x epoch_length x number of epochs)
	title_of_run = title % (identifier + 1)
	hfSEP = load(hfsep_path % (identifier + 1))
	noise = load(noise_path % (identifier + 1))

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
#		[np.concatenate((hfSEP.reshape(-1, hfSEP.shape[-1]), noise.reshape(-1, noise.shape[-1])), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), title_of_run + '_all_channels_flattened'],// I have no idea why, but this singular error still occurs
		[np.concatenate((hfSEP_CSP_0, noise_CSP_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), csp_title],
		[np.concatenate((ccar_filt_hfSEP_0, ccar_filt_noise_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), ccar_title],
		[np.concatenate((hil_extracted_CSP_hfSEP, hil_extracted_CSP_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_csp_title],
		[np.concatenate((hil_extracted_ccar_hfSEP, hil_extracted_ccar_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_ccar_title]
	]

# ToDo: Train without class imbalances, data prep to have class-imbalance data ready of 10noise:1hfSEP, train with class imbalances

### Loop over the different data-modalities and compute the confusion-matrix for each
## load the data, start with k0010, then go the range until k009 create a reproducible datasplit and shuffle the samples then
confusion_matrices = []

for i in range(10):

	idx = i

	for hfsep_dat, noise_dat, title in base_modalities_per_subject:

		### 04_ToDo: Classify all the 176 modalities using ELM
		### Load data of k003, to check on performance and compare w. results pusblished
		workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_dat, noise_dat, title, idx)
		
		for data, labels, run_title in workload:
			print('run_title: %s' % run_title)
			print(data.shape)

			### Shuffle and split data // .T is required to switch back to shape of (trial x feature)
			shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat)
			print(shuffled_data.shape)
			X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat)
			print(X_train.shape)

			## Some model definition
			elm = meet.elm.ClassELM()

			### Train ELM on the extracted features using cv for hyperparameter and then classify test-samples
			elm.cv(X_train, y_train, folds=5)
			if elm.istrained:
				print('Succesfully trained the ELM. Now work on gaining insights and getting performance metrics.')
				print('Performing classification on the test-data now, after optimizing the randomly initialized ELM. Result:')
				confusion_matrix_after_random_initialization = meet.elm.get_conf_matrix(y_test, elm.classify(X_test))
				print(confusion_matrix_after_random_initialization)
				confusion_matrices.append((run_title, confusion_matrix_after_random_initialization))
				with open('/media/christoph/Volume/Masterthesis/elm_models_trained/classes_balanced/elm_%s' % run_title.replace('/', '-'), 'wb') as target_file: 
				    pickle.dump(elm, target_file)

ssd_modalities = [
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_under_100_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_under_100_kx_noise.npy', 'k00%d_ssd_filt_intrplt_filt_under_100'] , 
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_100_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_100_kx_noise.npy', 'k00%d_ssd_filt_intrplt_filt_over_100'] , 
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_400_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_400_kx_noise.npy', 'k00%d_ssd_filt_intrplt_filt_over_400'] , 
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_500_900_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_500_900_kx_noise.npy', 'k00%d_ssd_filt_intrplt_filt_500_900']
]

for i in range(10):

	idx = i

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

			## Some model definition
			elm = meet.elm.ClassELM()

			### Train ELM on the extracted features using cv for hyperparameter and then classify test-samples
			elm.cv(X_train, y_train, folds=5)
			if elm.istrained:
				print('Succesfully trained the ELM. Now work on gaining insights and getting performance metrics.')
				print('Performing classification on the test-data now, after optimizing the randomly initialized ELM. Result:')
				confusion_matrix_after_random_initialization = meet.elm.get_conf_matrix(y_test, elm.classify(X_test))
				print(confusion_matrix_after_random_initialization)
				confusion_matrices.append((run_title, confusion_matrix_after_random_initialization))
				with open('/media/christoph/Volume/Masterthesis/elm_models_trained/classes_balanced/elm_%s' % run_title.replace('/', '-'), 'wb') as target_file: 
				    pickle.dump(elm, target_file)

from numpy import save
print(confusion_matrices)

save('/media/christoph/Volume/Masterthesis/elm_models_trained/balanced_classes_confusion_matrices/confusion_matrices_run_01_on_data_preprocessed_final', confusion_matrices)