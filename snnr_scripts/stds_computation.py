for idx, hfsep_data, noise_data, level in [(0, epchd_14_25_ms_intrplt_filt_500_900_kx, epchd_30_60_ms_intrplt_filt_500_900_kx_baseline, '500Hz_900Hz')]:

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

		stds_slid.append([std_over_time_cp5_minus_fz_pos_slid, std_over_time_cp5_minus_fz_neg_slid])
		stds_jump.append([std_over_time_cp5_minus_fz_pos_jump, std_over_time_cp5_minus_fz_neg_jump])

		std_of_stds_slid = np.std(std_over_time_cp5_minus_fz_pos_slid) / np.std(std_over_time_cp5_minus_fz_neg_slid)
		std_of_stds_jump = np.std(std_over_time_cp5_minus_fz_pos_jump) / np.std(std_over_time_cp5_minus_fz_neg_jump)
		variance_ratios.append([std_of_stds_slid, std_of_stds_jump])

# transform result-arrays   
variance_ratios = np.asarray(variance_ratios)
stds_slid = np.asarray(stds_slid)
stds_jump = np.asarray(stds_jump)