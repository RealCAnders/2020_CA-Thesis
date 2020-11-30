import sys
import datetime
import numpy as np
import scipy

from multiprocessing import Process
import matplotlib.pyplot as plt

from numpy import load, save
from scipy.ndimage import convolve1d

import meet

from matplotlib.font_manager import FontProperties 
font = FontProperties() 
font.set_family('serif') 
font.set_name('Times New Roman') 
font.set_style('oblique') 
font.set_size(15)

def normalize_min_max(to_be_normalized):  
    return (to_be_normalized - np.min(to_be_normalized)) / (np.max(to_be_normalized) - np.min(to_be_normalized)) 

# Falls Unklarheiten bestehen: Print der shapes!! Das hilft!!
def perform_snnr_var_rat_computation(triggers, data_path, intrplt_filt_500_900_kx_path, identifier, level, outlier_non_outlier):
	print('### ### ### ### ### ###')
	if outlier_non_outlier:
		print('Computing variance_ratios for outlier removed data of k00%d in level: %s' % ((identifier + 1), level))
	else: 
		print('Computing variance_ratios for non-outlier removed data of k00%d in level: %s' % ((identifier + 1), level))

	triggers_kx_data_combined = triggers
	intrplt_kx_data_combined = data_path
	intrplt_filt_500_900_kx = intrplt_filt_500_900_kx_path

	epchd_14_25_ms_intrplt_filt_500_900 = meet.epochEEG(intrplt_filt_500_900_kx, triggers_kx_data_combined, [100, 350])
	a, b, s = meet.spatfilt.CCAvReg(epchd_14_25_ms_intrplt_filt_500_900[:8,:,:])

	epchd_14_25_ms_intrplt_kx_data_combined = meet.epochEEG(intrplt_kx_data_combined, triggers_kx_data_combined, [100, 350])
	epchd_30_60_ms_intrplt_kx_data_combined_baseline = meet.epochEEG(intrplt_kx_data_combined, triggers_kx_data_combined, [400, 650])
	ccar_epchd_14_25_ms_intrplt_kx_data_combined = np.tensordot(a[:,0], epchd_14_25_ms_intrplt_kx_data_combined[:8,:,:], axes=(0,0))
	ccar_epchd_30_60_ms_intrplt_kx_data_combined_baseline = np.tensordot(a[:,0], epchd_30_60_ms_intrplt_kx_data_combined_baseline[:8,:,:], axes=(0,0))

	averaging_points = [1, 2, 5, 10, 15, 20, 30, 45, 60, 80, 100, 120, 150, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

	hfsep_data = ccar_epchd_14_25_ms_intrplt_kx_data_combined
	noise_data = ccar_epchd_30_60_ms_intrplt_kx_data_combined_baseline

	variance_ratios = []
	stds_slid = []
	stds_jump = []
	snnrs = []
	mean_stds_over_time_slid_hfsep = []
	mean_stds_over_time_slid_baseline = []
	std_of_stds_over_time_slid_hfsep = []
	std_of_stds_over_time_slid_baseline = []
	mean_stds_over_time_jump_hfsep = []
	mean_stds_over_time_jump_baseline = []
	std_of_stds_over_time_jump_hfsep = []
	std_of_stds_over_time_jump_baseline = []

	print('Performing SNNR-and-Var_Rat-computation at level [%s]' % level)

	## Das averaging wollen wir vor (!) der Hilbert-Transformation durchführen!
	for i in averaging_points:    
		# following decibel definition from: https://en.wikipedia.org/wiki/Signal-to-noise_ratio#Decibels    
		# multiply by 20 as written here: https://www.researchgate.net/post/What_is_the_best_method_to_calculate_the_S_N_ratio_from_potentiometric_data    
		if i == 1:

			ccar_slid_pos = convolve1d(hfsep_data, weights=np.ones(i) / i, axis=1)
			ccar_jump_pos = convolve1d(hfsep_data, weights=np.ones(i) / i, axis=1)
			ccar_slid_baseline = convolve1d(noise_data, weights=np.ones(i) / i, axis=1)
			ccar_jump_baseline = convolve1d(noise_data, weights=np.ones(i) / i, axis=1)

		else:

			ccar_slid_pos = convolve1d(hfsep_data, weights=np.ones(i) / i, axis=1)[:,int(i / 2):-int(i / 2)]
			ccar_jump_pos = convolve1d(hfsep_data, weights=np.ones(i) / i, axis=1)[:,int(i / 2):-int(i / 2):i]
			ccar_slid_baseline = convolve1d(noise_data, weights=np.ones(i) / i, axis=1)[:,int(i / 2):-int(i / 2)]
			ccar_jump_baseline = convolve1d(noise_data, weights=np.ones(i) / i, axis=1)[:,int(i / 2):-int(i / 2):i]

		# current shape: time_in_ccar_filt_trial x trial // e.g. (110 x 5940)
		
		## variance-ratio computations
		# STD je trial --> Power je trial (Montage CP5-FZ macht keinen Sinn mehr, da nun CCAr-gefiltert nur 1 Kanal)
		std_over_time_pos_slid = np.std(ccar_slid_pos, axis=0)
		std_over_time_neg_slid = np.std(ccar_slid_baseline, axis=0)
		std_over_time_pos_jump = np.std(ccar_jump_pos, axis=0)
		std_over_time_neg_jump = np.std(ccar_jump_baseline, axis=0)

		stds_slid.append([std_over_time_pos_slid, std_over_time_neg_slid])
		stds_jump.append([std_over_time_pos_jump, std_over_time_neg_jump])
		mean_stds_over_time_slid_hfsep.append(np.mean(std_over_time_pos_slid))
		mean_stds_over_time_slid_baseline.append(np.mean(std_over_time_neg_slid))
		std_of_stds_over_time_slid_hfsep.append(np.std(std_over_time_pos_slid))
		std_of_stds_over_time_slid_baseline.append(np.std(std_over_time_neg_slid))
		mean_stds_over_time_jump_hfsep.append(np.mean(std_over_time_pos_jump))
		mean_stds_over_time_jump_baseline.append(np.mean(std_over_time_neg_jump))
		std_of_stds_over_time_jump_hfsep.append(np.std(std_over_time_pos_jump))
		std_of_stds_over_time_jump_baseline.append(np.std(std_over_time_neg_jump))

		var_rat_slid = np.std(std_over_time_pos_slid) / np.std(std_over_time_neg_slid)
		var_rat_jump = np.std(std_over_time_pos_jump) / np.std(std_over_time_neg_jump)
		variance_ratios.append([var_rat_slid, var_rat_jump])

		## snnr-computations
		hil_ccar_slid_pos = scipy.signal.hilbert(ccar_slid_pos, axis=0)
		hil_ccar_jump_pos = scipy.signal.hilbert(ccar_jump_pos, axis=0)
		hil_ccar_slid_baseline = scipy.signal.hilbert(ccar_slid_baseline, axis=0)
		hil_ccar_jump_baseline = scipy.signal.hilbert(ccar_jump_baseline, axis=0)

		avg_amp_env_over_pos_slid = np.abs(hil_ccar_slid_pos).mean(1)
		avg_amp_env_over_baseline_slid = np.abs(hil_ccar_slid_baseline).mean(1)

		if hil_ccar_jump_pos.shape[1] > 1:
			avg_amp_env_over_pos_jump = np.abs(hil_ccar_jump_pos).mean(1)
			avg_amp_env_over_baseline_jump = np.abs(hil_ccar_jump_baseline).mean(1)
		else:
			avg_amp_env_over_pos_jump = np.abs(hil_ccar_jump_pos)
			avg_amp_env_over_baseline_jump = np.abs(hil_ccar_jump_baseline)

		max_avg_amp_hfseps_slid = avg_amp_env_over_pos_slid.max()
		max_avg_amp_hfseps_jump = avg_amp_env_over_pos_jump.max()
		mean_avg_amp_baseline_slid = avg_amp_env_over_baseline_slid.mean()
		mean_avg_amp_baseline_jump = avg_amp_env_over_baseline_jump.mean()

		# append snnr_slid vs. snnr_jump
		snnrs.append([(max_avg_amp_hfseps_slid / mean_avg_amp_baseline_slid), (max_avg_amp_hfseps_jump / mean_avg_amp_baseline_jump)])		

	variance_ratios = np.asarray(variance_ratios)
	stds_slid = np.asarray(stds_slid)
	stds_jump = np.asarray(stds_jump)
	snnrs = np.asarray(snnrs)
	mean_stds_over_time_slid_hfsep = np.asarray(mean_stds_over_time_slid_hfsep)
	mean_stds_over_time_slid_baseline = np.asarray(mean_stds_over_time_slid_baseline)
	std_of_stds_over_time_slid_hfsep = np.asarray(std_of_stds_over_time_slid_hfsep)
	std_of_stds_over_time_slid_baseline = np.asarray(std_of_stds_over_time_slid_baseline)
	mean_stds_over_time_jump_hfsep = np.asarray(mean_stds_over_time_jump_hfsep)
	mean_stds_over_time_jump_baseline = np.asarray(mean_stds_over_time_jump_baseline)
	std_of_stds_over_time_jump_hfsep = np.asarray(std_of_stds_over_time_jump_hfsep)
	std_of_stds_over_time_jump_baseline = np.asarray(std_of_stds_over_time_jump_baseline)

	save('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_variance_ratios.npy' % ((identifier + 1), level), variance_ratios)
	save('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_stds_slid.npy' % ((identifier + 1), level), stds_slid)
	save('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_stds_jump.npy' % ((identifier + 1), level), stds_jump)
	save('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_snnrs.npy' % ((identifier + 1), level), snnrs)
	save('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_mean_stds_over_time_slid_hfsep.npy' % ((identifier + 1), level), mean_stds_over_time_slid_hfsep)
	save('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_mean_stds_over_time_slid_baseline.npy' % ((identifier + 1), level), mean_stds_over_time_slid_baseline)
	save('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_std_of_stds_over_time_slid_hfsep.npy' % ((identifier + 1), level), std_of_stds_over_time_slid_hfsep)
	save('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_std_of_stds_over_time_slid_baseline.npy' % ((identifier + 1), level), std_of_stds_over_time_slid_baseline)
	save('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_mean_stds_over_time_jump_hfsep.npy' % ((identifier + 1), level), mean_stds_over_time_jump_hfsep)
	save('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_mean_stds_over_time_jump_baseline.npy' % ((identifier + 1), level), mean_stds_over_time_jump_baseline)
	save('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_std_of_stds_over_time_jump_hfsep.npy' % ((identifier + 1), level), std_of_stds_over_time_jump_hfsep)
	save('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d/%s_std_of_stds_over_time_jump_baseline.npy' % ((identifier + 1), level), std_of_stds_over_time_jump_baseline)

	fig = plt.figure(figsize=(12, 8))
	ax = fig.add_subplot(111)
	lin0 = plt.plot(averaging_points, snnrs[:,0], linestyle='-', color='black', label='mean_snnr', linewidth=3) 
	ax.set_ylabel('scale for SNNR', fontproperties=font)
	ax.set_ylim(0, 90)
	ax2 = plt.twinx()
	lin1 = ax2.plot(averaging_points, normalize_min_max(variance_ratios[:,0]), linestyle=':', color='black', label='var_rats', linewidth=3) 
	lin2 = ax2.plot(averaging_points, normalize_min_max(mean_stds_over_time_slid_hfsep), linestyle=':', color='#7a8041', label='mean_stds_hfSEP', linewidth=3) 
	lin3 = ax2.plot(averaging_points, normalize_min_max(mean_stds_over_time_slid_baseline), linestyle=':', color='#3503ff', label='mean_stds_baseline', linewidth=3) 
	lin4 = ax2.plot(averaging_points, normalize_min_max(std_of_stds_over_time_slid_hfsep), linestyle='-', color='#7a8041', label='std_of_stds_hfSEP', linewidth=3) 
	lin5 = ax2.plot(averaging_points, normalize_min_max(std_of_stds_over_time_slid_baseline), linestyle='-', color='#3503ff', label='std_of_stds_baseline', linewidth=3) 
	ax2.set_ylabel('normalized STDs-Scale for STDs and var-rats [0,1]', fontproperties=font)
	plt.title('K00%d %s filt, CCAr-Hil, snnr and var-rat showcase;' % ((identifier + 1), level), fontproperties=font)
	plt.xscale('log')
	ax.set_xlabel('# of trials used for averaging', fontproperties=font)
	lns = lin0 + lin1 + lin2 + lin3 + lin4 + lin5
	lbls = [l.get_label() for l in lns]
	ax.legend(lns, lbls, loc=9, fontsize=15) 
	plt.savefig('/home/christoph/Desktop/Beginning_October_Work/final_snnr_var_rat_compuatation/k00%d_plot_%s' % ((identifier + 1), level), dpi=300)
	plt.close('all')


trigger_dataset = ['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined.npy', 'triggers']
modalities_non_outlier_removed = [
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_kx_data_combined.npy', 'data_combined'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_under_100_kx.npy', 'under_100Hz'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_100_kx.npy', 'over_100Hz'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_400_kx.npy', 'over_400Hz'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy', '500Hz_900Hz']
]

trigger_dataset_out_rej = ['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/triggers_identified_after_rejection.npy', 'triggers_out_rej']
modalities_outlier_removed = [
	['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/kx_data_intrplt_outliers_rejected.npy', 'data_combined_out_rej'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/intrplt_filt_under_100_kx_out_rej.npy', 'under_100Hz_out_rej'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/intrplt_filt_over_100_kx_out_rej.npy', 'over_100Hz_out_rej'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/intrplt_filt_over_400_kx_out_rej.npy', 'over_400Hz_out_rej'],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/intrplt_filt_500_900_kx_out_rej.npy', '500Hz_900Hz_out_rej']
]

for identifier in np.arange(10):
	processes = []
	print('Now (%s) preparing work on non-outlier removed datasets' % str(datetime.datetime.now()))

	triggers_kx_data_combined = load(trigger_dataset[0] % (identifier + 1))
	intrplt_filt_500_900_kx = load(modalities_non_outlier_removed[-1][0] % (identifier + 1))
	for modality in modalities_non_outlier_removed:
		path, level = modality
#		triggers, data_path, intrplt_filt_500_900_kx_path, identifier, level, outlier_non_outlier
		intrplt_kx_data_combined = load(path % (identifier + 1))
		p = Process(target=perform_snnr_var_rat_computation, args=(triggers_kx_data_combined, intrplt_kx_data_combined, intrplt_filt_500_900_kx, identifier, level, False))
		processes.append(p)	

	triggers_kx_data_combined_out_rej = load(trigger_dataset_out_rej[0] % (identifier + 1))
	intrplt_filt_500_900_kx_out_rej = load(modalities_outlier_removed[-1][0] % (identifier + 1))
	for modality in modalities_outlier_removed:
		path, level = modality
		intrplt_kx_data_combined_out_rej = load(path % (identifier + 1))
		p = Process(target=perform_snnr_var_rat_computation, args=(triggers_kx_data_combined_out_rej, intrplt_kx_data_combined_out_rej, intrplt_filt_500_900_kx_out_rej, identifier, level, True, ))
		processes.append(p)	

	print('Finished data loading by (%s), going to start the work now...' % str(datetime.datetime.now()))

	for p in processes:
		p.start()

	for p in processes:
		p.join()

	print('Finished work by ' + str(datetime.datetime.now()))

##-#for identifier in np.arange(10):
##-#	print('### ### ### ### ### ###')
##-#	print('Computing variance_ratios for non-outlier removed data of k00%d' % (identifier + 1))
##-#	triggers_kx_data_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/triggers_identified_after_rejection.npy' % (identifier + 1))
##-#	intrplt_kx_data_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/kx_data_intrplt_outliers_rejected.npy' % (identifier + 1))
##-#	intrplt_filt_under_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/intrplt_filt_under_100_kx_out_rej.npy' % (identifier + 1))
##-#	intrplt_filt_over_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/intrplt_filt_over_100_kx_out_rej.npy' % (identifier + 1))
##-#	intrplt_filt_over_400_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/intrplt_filt_over_400_kx_out_rej.npy' % (identifier + 1))
##-#	intrplt_filt_500_900_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/intrplt_filt_500_900_kx_out_rej.npy' % (identifier + 1))
##-#
##-#	epchd_14_25_ms_intrplt_filt_500_900 = meet.epochEEG(intrplt_filt_500_900_kx, triggers_kx_data_combined, [100, 350])
##-#	a, b, s = meet.spatfilt.CCAvReg(epchd_14_25_ms_intrplt_filt_500_900[:8,:,:])
##-#
##-#	epchd_14_25_ms_intrplt_kx_data_combined = meet.epochEEG(intrplt_kx_data_combined, triggers_kx_data_combined, [100, 350])
##-#	epchd_14_25_ms_intrplt_filt_under_100_kx = meet.epochEEG(intrplt_filt_under_100_kx, triggers_kx_data_combined, [100, 350])
##-#	epchd_14_25_ms_intrplt_filt_over_100_kx = meet.epochEEG(intrplt_filt_over_100_kx, triggers_kx_data_combined, [100, 350])
##-#	epchd_14_25_ms_intrplt_filt_over_400_kx = meet.epochEEG(intrplt_filt_over_400_kx, triggers_kx_data_combined, [100, 350])
##-#	epchd_14_25_ms_intrplt_filt_500_900_kx = meet.epochEEG(intrplt_filt_500_900_kx, triggers_kx_data_combined, [100, 350])
##-#
##-#	epchd_30_60_ms_intrplt_kx_data_combined_baseline = meet.epochEEG(intrplt_kx_data_combined, triggers_kx_data_combined, [400, 650])
##-#	epchd_30_60_ms_intrplt_filt_under_100_kx_baseline = meet.epochEEG(intrplt_filt_under_100_kx, triggers_kx_data_combined, [400, 650])
##-#	epchd_30_60_ms_intrplt_filt_over_100_kx_baseline = meet.epochEEG(intrplt_filt_over_100_kx, triggers_kx_data_combined, [400, 650])
##-#	epchd_30_60_ms_intrplt_filt_over_400_kx_baseline = meet.epochEEG(intrplt_filt_over_400_kx, triggers_kx_data_combined, [400, 650])
##-#	epchd_30_60_ms_intrplt_filt_500_900_kx_baseline = meet.epochEEG(intrplt_filt_500_900_kx, triggers_kx_data_combined, [400, 650])
##-#
##-#	ccar_epchd_14_25_ms_intrplt_kx_data_combined = np.tensordot(a[:,0], epchd_14_25_ms_intrplt_kx_data_combined[:8,:,:], axes=(0,0))
##-#	ccar_epchd_14_25_ms_intrplt_filt_under_100_kx = np.tensordot(a[:,0], epchd_14_25_ms_intrplt_filt_under_100_kx[:8,:,:], axes=(0,0))
##-#	ccar_epchd_14_25_ms_intrplt_filt_over_100_kx = np.tensordot(a[:,0], epchd_14_25_ms_intrplt_filt_over_100_kx[:8,:,:], axes=(0,0))
##-#	ccar_epchd_14_25_ms_intrplt_filt_over_400_kx = np.tensordot(a[:,0], epchd_14_25_ms_intrplt_filt_over_400_kx[:8,:,:], axes=(0,0))
##-#	ccar_epchd_14_25_ms_intrplt_filt_500_900_kx = np.tensordot(a[:,0], epchd_14_25_ms_intrplt_filt_500_900_kx[:8,:,:], axes=(0,0))
##-#
##-#	ccar_epchd_30_60_ms_intrplt_kx_data_combined_baseline = np.tensordot(a[:,0], epchd_30_60_ms_intrplt_kx_data_combined_baseline[:8,:,:], axes=(0,0))
##-#	ccar_epchd_30_60_ms_intrplt_filt_under_100_kx_baseline = np.tensordot(a[:,0], epchd_30_60_ms_intrplt_filt_under_100_kx_baseline[:8,:,:], axes=(0,0))
##-#	ccar_epchd_30_60_ms_intrplt_filt_over_100_kx_baseline = np.tensordot(a[:,0], epchd_30_60_ms_intrplt_filt_over_100_kx_baseline[:8,:,:], axes=(0,0))
##-#	ccar_epchd_30_60_ms_intrplt_filt_over_400_kx_baseline = np.tensordot(a[:,0], epchd_30_60_ms_intrplt_filt_over_400_kx_baseline[:8,:,:], axes=(0,0))
##-#	ccar_epchd_30_60_ms_intrplt_filt_500_900_kx_baseline = np.tensordot(a[:,0], epchd_30_60_ms_intrplt_filt_500_900_kx_baseline[:8,:,:], axes=(0,0))
##-#
##-#	## Diesen Teil hier jeweils mit Jumping-Windows berechnen!
##-#
##-#	##--##variance_ratios = [[], [], [], [], []] 
##-#	##--##stds_slid = [[], [], [], [], []] 
##-#	##--##stds_jump = [[], [], [], [], []]
##-#	##--##snnrs = [[], [], [], [], []]
##-#
##-#	averaging_points = [1, 2, 5, 10, 15, 20, 30, 45, 60, 80, 100, 120, 150, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
##-#
##-#	for idx, hfsep_data, noise_data, level in [(0, ccar_epchd_14_25_ms_intrplt_kx_data_combined, ccar_epchd_30_60_ms_intrplt_kx_data_combined_baseline, 'data_combined'), 
##-#		(1, ccar_epchd_14_25_ms_intrplt_filt_under_100_kx, ccar_epchd_30_60_ms_intrplt_filt_under_100_kx_baseline, 'under_100Hz'),
##-#		(2, ccar_epchd_14_25_ms_intrplt_filt_over_100_kx, ccar_epchd_30_60_ms_intrplt_filt_over_100_kx_baseline, 'over_100Hz'),
##-#		(3, ccar_epchd_14_25_ms_intrplt_filt_over_400_kx, ccar_epchd_30_60_ms_intrplt_filt_over_400_kx_baseline, 'over_400Hz'),
##-#		(4, ccar_epchd_14_25_ms_intrplt_filt_500_900_kx, ccar_epchd_30_60_ms_intrplt_filt_500_900_kx_baseline, '500Hz_900Hz')]:
##-#
##-#	#idx, hfsep_data, noise_data, level = (4, ccar_epchd_14_25_ms_intrplt_filt_500_900_kx, ccar_epchd_30_60_ms_intrplt_filt_500_900_kx_baseline, '500Hz_900Hz')
		##-#
##-#		variance_ratios = []
##-#		stds_slid = []
##-#		stds_jump = []
##-#		snnrs = []
##-#		mean_stds_over_time_slid_hfsep = []
##-#		mean_stds_over_time_slid_baseline = []
##-#		std_of_stds_over_time_slid_hfsep = []
##-#		std_of_stds_over_time_slid_baseline = []
##-#		mean_stds_over_time_jump_hfsep = []
##-#		mean_stds_over_time_jump_baseline = []
##-#		std_of_stds_over_time_jump_hfsep = []
##-#		std_of_stds_over_time_jump_baseline = []
##-#
##-#		print('Performing SNNR-and-Var_Rat-computation at level [%s]' % level)
##-#
##-#		## Das averaging wollen wir vor (!) der Hilbert-Transformation durchführen!
##-#		for i in averaging_points:    
##-#			# following decibel definition from: https://en.wikipedia.org/wiki/Signal-to-noise_ratio#Decibels    
##-#			# multiply by 20 as written here: https://www.researchgate.net/post/What_is_the_best_method_to_calculate_the_S_N_ratio_from_potentiometric_data    
##-#			if i == 1:
##-#
##-#				ccar_slid_pos = convolve1d(hfsep_data, weights=np.ones(i) / i, axis=1)
##-#				ccar_jump_pos = convolve1d(hfsep_data, weights=np.ones(i) / i, axis=1)
##-#				ccar_slid_baseline = convolve1d(noise_data, weights=np.ones(i) / i, axis=1)
##-#				ccar_jump_baseline = convolve1d(noise_data, weights=np.ones(i) / i, axis=1)
##-#
##-#			else:
##-#
##-#				ccar_slid_pos = convolve1d(hfsep_data, weights=np.ones(i) / i, axis=1)[:,int(i / 2):-int(i / 2)]
##-#				ccar_jump_pos = convolve1d(hfsep_data, weights=np.ones(i) / i, axis=1)[:,int(i / 2):-int(i / 2):i]
##-#				ccar_slid_baseline = convolve1d(noise_data, weights=np.ones(i) / i, axis=1)[:,int(i / 2):-int(i / 2)]
##-#				ccar_jump_baseline = convolve1d(noise_data, weights=np.ones(i) / i, axis=1)[:,int(i / 2):-int(i / 2):i]
##-#
##-#			# current shape: time_in_ccar_filt_trial x trial // e.g. (110 x 5940)
			##-#
##-#			## variance-ratio computations
##-#			# STD je trial --> Power je trial (Montage CP5-FZ macht keinen Sinn mehr, da nun CCAr-gefiltert nur 1 Kanal)
##-#			std_over_time_pos_slid = np.std(ccar_slid_pos, axis=0)
##-#			std_over_time_neg_slid = np.std(ccar_slid_baseline, axis=0)
##-#			std_over_time_pos_jump = np.std(ccar_jump_pos, axis=0)
##-#			std_over_time_neg_jump = np.std(ccar_jump_baseline, axis=0)
##-#
##-#		#	stds_slid[idx].append([std_over_time_pos_slid, std_over_time_neg_slid])
##-#		#	stds_jump[idx].append([std_over_time_pos_jump, std_over_time_neg_jump])
##-#			stds_slid.append([std_over_time_pos_slid, std_over_time_neg_slid])
##-#			stds_jump.append([std_over_time_pos_jump, std_over_time_neg_jump])
##-#			mean_stds_over_time_slid_hfsep.append(np.mean(std_over_time_pos_slid))
##-#			mean_stds_over_time_slid_baseline.append(np.mean(std_over_time_neg_slid))
##-#			std_of_stds_over_time_slid_hfsep.append(np.std(std_over_time_pos_slid))
##-#			std_of_stds_over_time_slid_baseline.append(np.std(std_over_time_neg_slid))
##-#			mean_stds_over_time_jump_hfsep.append(np.mean(std_over_time_pos_jump))
##-#			mean_stds_over_time_jump_baseline.append(np.mean(std_over_time_neg_jump))
##-#			std_of_stds_over_time_jump_hfsep.append(np.std(std_over_time_pos_jump))
##-#			std_of_stds_over_time_jump_baseline.append(np.std(std_over_time_neg_jump))
##-#
##-#			# Backup / How it roughly would have been w. old snnr-computation
##-#			# snnr_slid = np.mean(std_over_time_cp5_minus_fz_pos_slid / std_over_time_cp5_minus_fz_neg_slid)
##-#			# snnr_pos = np.mean(std_over_time_cp5_minus_fz_pos_jump / std_over_time_cp5_minus_fz_neg_jump)
##-#			var_rat_slid = np.std(std_over_time_pos_slid) / np.std(std_over_time_neg_slid)
##-#			var_rat_jump = np.std(std_over_time_pos_jump) / np.std(std_over_time_neg_jump)
##-#			variance_ratios.append([var_rat_slid, var_rat_jump])
##-#		#	variance_ratios[idx].append([var_rat_slid, var_rat_jump])
##-#
##-#			## snnr-computations
##-#			# print(ccar_slid_pos.shape)
##-#			hil_ccar_slid_pos = scipy.signal.hilbert(ccar_slid_pos, axis=0)
##-#			hil_ccar_jump_pos = scipy.signal.hilbert(ccar_jump_pos, axis=0)
##-#			hil_ccar_slid_baseline = scipy.signal.hilbert(ccar_slid_baseline, axis=0)
##-#			hil_ccar_jump_baseline = scipy.signal.hilbert(ccar_jump_baseline, axis=0)
##-#
##-#			avg_amp_env_over_pos_slid = np.abs(hil_ccar_slid_pos).mean(1)
##-#			avg_amp_env_over_baseline_slid = np.abs(hil_ccar_slid_baseline).mean(1)
##-#
##-#			if hil_ccar_jump_pos.shape[1] > 1:
##-#				avg_amp_env_over_pos_jump = np.abs(hil_ccar_jump_pos).mean(1)
##-#				avg_amp_env_over_baseline_jump = np.abs(hil_ccar_jump_baseline).mean(1)
##-#			else:
##-#				avg_amp_env_over_pos_jump = np.abs(hil_ccar_jump_pos)
##-#				avg_amp_env_over_baseline_jump = np.abs(hil_ccar_jump_baseline)
##-#
##-#			max_avg_amp_hfseps_slid = avg_amp_env_over_pos_slid.max()
##-#			max_avg_amp_hfseps_jump = avg_amp_env_over_pos_jump.max()
##-#			mean_avg_amp_baseline_slid = avg_amp_env_over_baseline_slid.mean()
##-#			mean_avg_amp_baseline_jump = avg_amp_env_over_baseline_jump.mean()
##-#
##-#			# append snnr_slid vs. snnr_jump
##-#		#	snnrs[idx].append([(max_avg_amp_hfseps_slid / mean_avg_amp_baseline_slid), (max_avg_amp_hfseps_jump / mean_avg_amp_baseline_jump)])		
##-#			snnrs.append([(max_avg_amp_hfseps_slid / mean_avg_amp_baseline_slid), (max_avg_amp_hfseps_jump / mean_avg_amp_baseline_jump)])		
##-#
##-#		#print('Computed SNNRs:')
##-#		#print(snnrs[idx])
##-#
##-#		variance_ratios = np.asarray(variance_ratios)
##-#		stds_slid = np.asarray(stds_slid)
##-#		stds_jump = np.asarray(stds_jump)
##-#		snnrs = np.asarray(snnrs)
##-#		mean_stds_over_time_slid_hfsep = np.asarray(mean_stds_over_time_slid_hfsep)
##-#		mean_stds_over_time_slid_baseline = np.asarray(mean_stds_over_time_slid_baseline)
##-#		std_of_stds_over_time_slid_hfsep = np.asarray(std_of_stds_over_time_slid_hfsep)
##-#		std_of_stds_over_time_slid_baseline = np.asarray(std_of_stds_over_time_slid_baseline)
##-#		mean_stds_over_time_jump_hfsep = np.asarray(mean_stds_over_time_jump_hfsep)
##-#		mean_stds_over_time_jump_baseline = np.asarray(mean_stds_over_time_jump_baseline)
##-#		std_of_stds_over_time_jump_hfsep = np.asarray(std_of_stds_over_time_jump_hfsep)
##-#		std_of_stds_over_time_jump_baseline = np.asarray(std_of_stds_over_time_jump_baseline)
##-#
##-#		save('/home/christoph/Desktop/Beginning_October_Work/newest_snnr_var_rat_computed/k00%d/outliers_rejected_%s_variance_ratios.npy' % ((identifier + 1), level), variance_ratios)
##-#		save('/home/christoph/Desktop/Beginning_October_Work/newest_snnr_var_rat_computed/k00%d/outliers_rejected_%s_stds_slid.npy' % ((identifier + 1), level), stds_slid)
##-#		save('/home/christoph/Desktop/Beginning_October_Work/newest_snnr_var_rat_computed/k00%d/outliers_rejected_%s_stds_jump.npy' % ((identifier + 1), level), stds_jump)
##-#		save('/home/christoph/Desktop/Beginning_October_Work/newest_snnr_var_rat_computed/k00%d/outliers_rejected_%s_snnrs.npy' % ((identifier + 1), level), snnrs)
##-#		save('/home/christoph/Desktop/Beginning_October_Work/newest_snnr_var_rat_computed/k00%d/outliers_rejected_%s_mean_stds_over_time_slid_hfsep.npy' % ((identifier + 1), level), mean_stds_over_time_slid_hfsep)
##-#		save('/home/christoph/Desktop/Beginning_October_Work/newest_snnr_var_rat_computed/k00%d/outliers_rejected_%s_mean_stds_over_time_slid_baseline.npy' % ((identifier + 1), level), mean_stds_over_time_slid_baseline)
##-#		save('/home/christoph/Desktop/Beginning_October_Work/newest_snnr_var_rat_computed/k00%d/outliers_rejected_%s_std_of_stds_over_time_slid_hfsep.npy' % ((identifier + 1), level), std_of_stds_over_time_slid_hfsep)
##-#		save('/home/christoph/Desktop/Beginning_October_Work/newest_snnr_var_rat_computed/k00%d/outliers_rejected_%s_std_of_stds_over_time_slid_baseline.npy' % ((identifier + 1), level), std_of_stds_over_time_slid_baseline)
##-#		save('/home/christoph/Desktop/Beginning_October_Work/newest_snnr_var_rat_computed/k00%d/outliers_rejected_%s_mean_stds_over_time_jump_hfsep.npy' % ((identifier + 1), level), mean_stds_over_time_jump_hfsep)
##-#		save('/home/christoph/Desktop/Beginning_October_Work/newest_snnr_var_rat_computed/k00%d/outliers_rejected_%s_mean_stds_over_time_jump_baseline.npy' % ((identifier + 1), level), mean_stds_over_time_jump_baseline)
##-#		save('/home/christoph/Desktop/Beginning_October_Work/newest_snnr_var_rat_computed/k00%d/outliers_rejected_%s_std_of_stds_over_time_jump_hfsep.npy' % ((identifier + 1), level), std_of_stds_over_time_jump_hfsep)
##-#		save('/home/christoph/Desktop/Beginning_October_Work/newest_snnr_var_rat_computed/k00%d/outliers_rejected_%s_std_of_stds_over_time_jump_baseline.npy' % ((identifier + 1), level), std_of_stds_over_time_jump_baseline)
##-#
##-#		def normalize_min_max(to_be_normalized):  
##-#		    return (to_be_normalized - np.min(to_be_normalized)) / (np.max(to_be_normalized) - np.min(to_be_normalized)) 
##-#
##-#		fig = plt.figure(figsize=(12, 8))
##-#		ax = fig.add_subplot(111)
##-#		lin0 = plt.plot(averaging_points, snnrs[:,0], linestyle='-', color='black', label='mean_snnr', linewidth=3) 
##-#		ax.set_ylabel('scale for SNNR', fontproperties=font)
##-#		ax.set_ylim(0, 90)
##-#		ax2 = plt.twinx()
##-#		lin1 = ax2.plot(averaging_points, normalize_min_max(variance_ratios[:,0]), linestyle=':', color='black', label='var_rats', linewidth=3) 
##-#		lin2 = ax2.plot(averaging_points, normalize_min_max(mean_stds_over_time_slid_hfsep), linestyle=':', color='#7a8041', label='mean_stds_hfSEP', linewidth=3) 
##-#		lin3 = ax2.plot(averaging_points, normalize_min_max(mean_stds_over_time_slid_baseline), linestyle=':', color='#3503ff', label='mean_stds_baseline', linewidth=3) 
##-#		lin4 = ax2.plot(averaging_points, normalize_min_max(std_of_stds_over_time_slid_hfsep), linestyle='-', color='#7a8041', label='std_of_stds_hfSEP', linewidth=3) 
##-#		lin5 = ax2.plot(averaging_points, normalize_min_max(std_of_stds_over_time_slid_baseline), linestyle='-', color='#3503ff', label='std_of_stds_baseline', linewidth=3) 
##-#		ax2.set_ylabel('normalized STDs-Scale for STDs and var-rats [0,1]', fontproperties=font)
##-#		plt.title('K00%d %s filt, CCAr-Hil, snnr and var-rat showcase;' % ((identifier + 1), level), fontproperties=font)
##-#		plt.xscale('log')
##-#		ax.set_xlabel('# of trials used for averaging', fontproperties=font)
##-#		lns = lin0 + lin1 + lin2 + lin3 + lin4 + lin5
##-#		lbls = [l.get_label() for l in lns]
##-#		ax.legend(lns, lbls, loc=9, fontsize=15) 
##-#		plt.savefig('/home/christoph/Desktop/Beginning_October_Work/newest_snnr_var_rat_computed/outliers_rejected_k00%d_plot_%s' % ((identifier + 1), level), dpi=300)
##-#		plt.close('all')