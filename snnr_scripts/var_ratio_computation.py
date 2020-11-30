import sys
import numpy as np
import scipy

from numpy import load, save
from scipy.ndimage import convolve1d

import meet

identifier = int(sys.argv[1])

print('### ### ### ### ### ###')
print('Computing variance_ratios for non-outlier removed data of k00%d' % (identifier + 1))
triggers_kx_data_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined.npy' % (identifier + 1))
intrplt_kx_data_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_kx_data_combined.npy' % (identifier + 1))
intrplt_filt_under_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_under_100_kx.npy' % (identifier + 1))
intrplt_filt_over_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_100_kx.npy' % (identifier + 1))
intrplt_filt_over_400_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_400_kx.npy' % (identifier + 1))
intrplt_filt_500_900_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy' % (identifier + 1))

epchd_14_25_ms_intrplt_filt_500_900 = meet.epochEEG(intrplt_filt_500_900_kx, triggers_kx_data_combined, [140, 250])
a, b, s = meet.spatfilt.CCAvReg(epchd_14_25_ms_intrplt_filt_500_900[:8,:,:])

epchd_14_25_ms_intrplt_kx_data_combined = meet.epochEEG(intrplt_kx_data_combined, triggers_kx_data_combined, [140, 250])
epchd_14_25_ms_intrplt_filt_under_100_kx = meet.epochEEG(intrplt_filt_under_100_kx, triggers_kx_data_combined, [140, 250])
epchd_14_25_ms_intrplt_filt_over_100_kx = meet.epochEEG(intrplt_filt_over_100_kx, triggers_kx_data_combined, [140, 250])
epchd_14_25_ms_intrplt_filt_over_400_kx = meet.epochEEG(intrplt_filt_over_400_kx, triggers_kx_data_combined, [140, 250])
epchd_14_25_ms_intrplt_filt_500_900_kx = meet.epochEEG(intrplt_filt_500_900_kx, triggers_kx_data_combined, [140, 250])

epchd_30_60_ms_intrplt_kx_data_combined_baseline = meet.epochEEG(intrplt_kx_data_combined, triggers_kx_data_combined, [300, 600])
epchd_30_60_ms_intrplt_filt_under_100_kx_baseline = meet.epochEEG(intrplt_filt_under_100_kx, triggers_kx_data_combined, [300, 600])
epchd_30_60_ms_intrplt_filt_over_100_kx_baseline = meet.epochEEG(intrplt_filt_over_100_kx, triggers_kx_data_combined, [300, 600])
epchd_30_60_ms_intrplt_filt_over_400_kx_baseline = meet.epochEEG(intrplt_filt_over_400_kx, triggers_kx_data_combined, [300, 600])
epchd_30_60_ms_intrplt_filt_500_900_kx_baseline = meet.epochEEG(intrplt_filt_500_900_kx, triggers_kx_data_combined, [300, 600])

## Diesen Teil hier jeweils mit Jumping-Windows berechnen!

variance_ratios = [[], [], [], [], []] 
stds_slid = [[], [], [], [], []] 
stds_jump = [[], [], [], [], []]

averaging_points = [1, 2, 5, 10, 15, 20, 30, 45, 60, 80, 100, 120, 150, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

for idx, hfsep_data, noise_data, level in [(0, epchd_14_25_ms_intrplt_kx_data_combined, epchd_30_60_ms_intrplt_kx_data_combined_baseline, 'data_combined'), 
	(1, epchd_14_25_ms_intrplt_filt_under_100_kx, epchd_30_60_ms_intrplt_filt_under_100_kx_baseline, 'under_100Hz'),
	(2, epchd_14_25_ms_intrplt_filt_over_100_kx, epchd_30_60_ms_intrplt_filt_over_100_kx_baseline, 'over_100Hz'),
	(3, epchd_14_25_ms_intrplt_filt_over_400_kx, epchd_30_60_ms_intrplt_filt_over_400_kx_baseline, 'over_400Hz'),
	(4, epchd_14_25_ms_intrplt_filt_500_900_kx, epchd_30_60_ms_intrplt_filt_500_900_kx_baseline, '500Hz_900Hz')]:

	print('Performing SNNR-computation at level [%s]' % level)

	for i in averaging_points:    
		# following decibel definition from: https://en.wikipedia.org/wiki/Signal-to-noise_ratio#Decibels    
		# multiply by 20 as written here: https://www.researchgate.net/post/What_is_the_best_method_to_calculate_the_S_N_ratio_from_potentiometric_data    
		if i == 1:

			sliding_windowed_pos = convolve1d(hfsep_data[:8,:,:], weights=np.ones(1), axis=2)[:,:,:]    
			jumping_windowed_pos = convolve1d(hfsep_data[:8,:,:], weights=np.ones(1), axis=2)[:,:,:]  
			sliding_windowed_noise = convolve1d(noise_data[:8,:,:], weights=np.ones(1), axis=2)[:,:,:]    
			jumping_windowed_noise = convolve1d(noise_data[:8,:,:], weights=np.ones(1), axis=2)[:,:,:]  

		else:

			jumping_windowed_pos = convolve1d(hfsep_data[:8,:,:], weights=np.ones(i) / i, axis=2)[:,:,int(i / 2):-int(i / 2):i]  
			sliding_windowed_pos = convolve1d(hfsep_data[:8,:,:], weights=np.ones(i) / i, axis=2)[:,:,int(i / 2):-int(i / 2)]    
			jumping_windowed_noise = convolve1d(noise_data[:8,:,:], weights=np.ones(i) / i, axis=2)[:,:,int(i / 2):-int(i / 2):i]
			sliding_windowed_noise = convolve1d(noise_data[:8,:,:], weights=np.ones(i) / i, axis=2)[:,:,int(i / 2):-int(i / 2)]

		# Swap from (chan, time_in_sample, sample) to (sample, chan, time_in_sample)
		sliding_windowed_pos = np.swapaxes(np.swapaxes(sliding_windowed_pos, 0, 1), 0, 2)   
		sliding_windowed_noise = np.swapaxes(np.swapaxes(sliding_windowed_noise, 0, 1), 0, 2)    
		jumping_windowed_pos = np.swapaxes(np.swapaxes(jumping_windowed_pos, 0, 1), 0, 2)  
		jumping_windowed_noise = np.swapaxes(np.swapaxes(jumping_windowed_noise, 0, 1), 0, 2)  

		std_over_time_cp5_minus_fz_pos_slid = np.std((sliding_windowed_pos[:,5] - sliding_windowed_pos[:,0]), axis=1)
		std_over_time_cp5_minus_fz_neg_slid = np.std((sliding_windowed_noise[:,5] - sliding_windowed_noise[:,0]), axis=1)
		std_over_time_cp5_minus_fz_pos_jump = np.std((jumping_windowed_pos[:,5] - jumping_windowed_pos[:,0]), axis=1)
		std_over_time_cp5_minus_fz_neg_jump = np.std((jumping_windowed_noise[:,5] - jumping_windowed_noise[:,0]), axis=1)

		stds_slid[idx].append([std_over_time_cp5_minus_fz_pos_slid, std_over_time_cp5_minus_fz_neg_slid])
		stds_jump[idx].append([std_over_time_cp5_minus_fz_pos_jump, std_over_time_cp5_minus_fz_neg_jump])

		std_of_stds_slid = np.std(std_over_time_cp5_minus_fz_pos_slid) / np.std(std_over_time_cp5_minus_fz_neg_slid)
		std_of_stds_jump = np.std(std_over_time_cp5_minus_fz_pos_jump) / np.std(std_over_time_cp5_minus_fz_neg_jump)
		variance_ratios[idx].append([std_of_stds_slid, std_of_stds_jump])

# transform result-arrays   
variance_ratios = np.asarray(variance_ratios)
stds_slid = np.asarray(stds_slid)
stds_jump = np.asarray(stds_jump)

print(variance_ratios.shape)
print(variance_ratios[0])

#print('variance_ratios:')
#print(variance_ratios)
#print('stds_slid:')
#print(stds_slid)
#print('stds_jump:')
#print(stds_jump)
save('/home/christoph/Desktop/End_Of_September_Work/variance_ratios/k00%d_variance_ratios.npy' % (identifier + 1), variance_ratios)
save('/home/christoph/Desktop/End_Of_September_Work/variance_ratios/k00%d_stds_slid.npy' % (identifier + 1), stds_slid)
save('/home/christoph/Desktop/End_Of_September_Work/variance_ratios/k00%d_stds_jump.npy' % (identifier + 1), stds_jump)

#	print('### ### ### ### ### ###')
#	print(variance_ratios)
#	print('### ### ### ### ### ###')
