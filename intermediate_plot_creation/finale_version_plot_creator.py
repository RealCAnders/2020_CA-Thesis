import sys 
import meet   
import numpy as np   
import matplotlib.pyplot as plt   
import scipy 
from scipy import signal   
from scipy.fft import fftshift   
from scipy.ndimage import convolve1d, convolve   
from numpy import save 
from numpy import load                                                                                                                                                                                    
from meet.spatfilt import CSP 
from matplotlib.font_manager import FontProperties
 
### ### ### ### ### 
### Definition: utility-function / global vars 
### ### ### ### ### 
offset = 1000 
s_rate = 10000 
stim_per_sec = 4 
out_rej_thresh_fz = [0.45, 0.5, 0.225, 0.6, 0.6, 0.4, 0.45, 0.75, 0.45, 2] 
out_rej_thresh_mean = [0.6, 0.415, 0.12, 0.75, 0.3, 0.3, 0.45, 0.45, 0.3, 1.5]                                                                                                                            
### ### ### ### ### 
### Data loading 
### ### ### ### ### 
hfSEP_win = [50, 450] 
noise_win = [-500, -100] 
intrplt_win = [-80, 30]                                                                                                                                                                                   

fig, axes = plt.subplot(5,2. sharey sharex)
for i in range(num_rows):
	axes[row, col_id].plot()...
	plt.text(0.95, 0.95, 'TEXT N = ....', horizontal_alignmetn=center, verticalalgi=center, transform=axes[row, col_id].transAxes)

def create_plots_for_thesis_of_subject(subject_id, path):
	data_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/kx_data_combined.npy' % subject_id, allow_pickle=True)                                                                  
	triggers_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined.npy' % subject_id, allow_pickle=True) 

	font = FontProperties()  
	font.set_family('serif')  
	font.set_name('Times New Roman')  
	font.set_size(20)        
	ticks = np.arange(-500, 600, 100)
	ticklabels = ['%d' % tick for tick in (ticks / 10)]

	# basic plot
	hfsep_around_artifact = meet.epochEEG(data_k3_combined, triggers_k3_combined, [-497, 503])
	std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
	ln0 = plt.plot(np.arange(-500, 500), (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1) - (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	ln1 = plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	ln2 = plt.plot(np.arange(-500, 500), (hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500]) - (hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500]).mean(), label='single-trial example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('voltage in μV', fontproperties=font)
	plt.ylim([-40, 40])
	#plt.title('FZ-CP5; artifact at 0ms, wideband signal, hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=20)
	lns, lbls = plt.gca().get_legend_handles_labels()
	lns = [lns[1], lns[2], lns[0]]
	lbls = [lbls[1], lbls[2], lbls[0]]
	plt.legend(lns, lbls, fontsize=20, loc='upper center', bbox_to_anchor=(0.5, 1.12), fancybox=True, shadow=True, ncol=3)
	# Reuse for old plots
	# plt.legend(fontsize=10, loc=2)
	plt.savefig(path % (subject_id, 'hfSEP_raw'), dpi=200)
	plt.close('all')


	### Plots per spectral filter modality
	intrplt_filt_under_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_under_100_kx.npy' % subject_id, allow_pickle=True)
	hfsep_around_artifact = meet.epochEEG(intrplt_filt_under_100_kx, triggers_k3_combined, [-500, 500])
	std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
	plt.plot(np.arange(-500, 500), (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1) - (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500] - (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1)[0], label='single-trial example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('voltage in μV', fontproperties=font)
	plt.title('FZ-CP5; data interpolated and IIR-filtered [f <= 100Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, 'intrplt_filt_under_100'), dpi=200)
	plt.close('all')


	intrplt_filt_over_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_100_kx.npy' % subject_id, allow_pickle=True)
	hfsep_around_artifact = meet.epochEEG(intrplt_filt_over_100_kx, triggers_k3_combined, [-500, 500])
	std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
	plt.plot(np.arange(-500, 500), (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1) - (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500], label='single-trial example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('voltage in μV', fontproperties=font)
	plt.title('FZ-CP5; data interpolated and IIR-filtered [f >= 100Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, 'intrplt_filt_over_100'), dpi=200)
	plt.close('all')


	intrplt_filt_over_400_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_400_kx.npy' % subject_id, allow_pickle=True)
	hfsep_around_artifact = meet.epochEEG(intrplt_filt_over_400_kx, triggers_k3_combined, [-500, 500])
	std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
	plt.plot(np.arange(-500, 500), (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1) - (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500], label='single-trial example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('voltage in μV', fontproperties=font)
	plt.title('FZ-CP5; data interpolated and IIR-filtered [f >= 400Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, 'intrplt_filt_over_400'), dpi=200)
	plt.close('all')


	intrplt_filt_500_900_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy' % subject_id, allow_pickle=True)
	hfsep_around_artifact = meet.epochEEG(intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500])
	std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
	plt.plot(np.arange(-500, 500), (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1) - (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500], label='single-trial example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('voltage in μV', fontproperties=font)
	plt.title('FZ-CP5; data interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, 'intrplt_filt_500_900'), dpi=200)
	plt.close('all')

	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
	ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
	hfsep_around_artifact = meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500])
	std_basic_prep = np.std(hfsep_around_artifact, axis=1)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact.mean(1) - hfsep_around_artifact.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact[:,3], label='single-trial example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	#-# plt.xscale('log')
	plt.ylim([-2.5, 2.5])
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('feature value in a.u.', fontproperties=font)
	plt.title('CCAr; interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, '500_900_CCAr'), dpi=200)
	plt.close('all')


	# ToDo: Do the averaging and the plot-creation after averaging now!
	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
	ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(10) / 10, axis=1)[:,int(10 / 2):-int(10 / 2):10]
	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	plt.ylim([-2.5, 2.5])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('feature value in a.u.', fontproperties=font)
	plt.title('10 trials sub-average CCAr; interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, '500_900_CCAr_10_subs'), dpi=200)
	plt.close('all')


	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
	ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(20) / 20, axis=1)[:,int(20 / 2):-int(20 / 2):20]
	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	plt.ylim([-2.5, 2.5])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('feature value in a.u.', fontproperties=font)
	plt.title('20 trials sub-average CCAr; interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, '500_900_CCAr_20_subs'), dpi=200)
	plt.close('all')


	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
	ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(500) / 500, axis=1)[:,int(500 / 2):-int(500 / 2):500]
	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	plt.ylim([-2.5, 2.5])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('feature value in a.u.', fontproperties=font)
	plt.title('500 trials sub-average CCAr; interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, '500_900_CCAr_500_subs'), dpi=200)
	plt.close('all')


	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
	csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
	hfsep_around_artifact = meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500])
	std_basic_prep = np.std(hfsep_around_artifact, axis=1)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact.mean(1) - hfsep_around_artifact.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact[:,3], label='single-trial example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	plt.ylim([-2.5, 2.5])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('feature value in a.u.', fontproperties=font)
	plt.title('CSP; interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, '500_900_CSP'), dpi=200)
	plt.close('all')


	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
	csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(10) / 10, axis=1)[:,int(10 / 2):-int(10 / 2):10]
	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	plt.ylim([-2.5, 2.5])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('feature value in a.u.', fontproperties=font)
	plt.title('10 trials sub-average CSP; interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, '500_900_CSP_10_subs'), dpi=200)
	plt.close('all')


	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
	csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(20) / 20, axis=1)[:,int(20 / 2):-int(20 / 2):20]
	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	plt.ylim([-2.5, 2.5])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('feature value in a.u.', fontproperties=font)
	plt.title('20 trials sub-average CSP; interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, '500_900_CSP_20_subs'), dpi=200)
	plt.close('all')


	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
	csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(500) / 500, axis=1)[:,int(500 / 2):-int(500 / 2):500]
	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	plt.ylim([-2.5, 2.5])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('feature value in a.u.', fontproperties=font)
	plt.title('500 trials sub-average CSP; interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, '500_900_CSP_500_subs'), dpi=200)
	plt.close('all')


	#SSD
	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx[:8], data_k3_combined[:8])
	ssd_filt_intrplt_filt_500_900_kx = np.tensordot(ssd_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ssd_filt_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(1) / 1, axis=1)[:,int(1):-int(1):1]
	std_basic_prep = np.std(meet.epochEEG(ssd_filt_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), axis=1)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='single-trial example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	plt.ylim([-0.2, 0.2])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('feature value in a.u.', fontproperties=font)
	plt.title('SSD; interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, '500_900_SSD'), dpi=200)
	plt.close('all')


	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx[:8], data_k3_combined[:8])
	ssd_filt_intrplt_filt_500_900_kx = np.tensordot(ssd_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ssd_filt_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(10) / 10, axis=1)[:,int(10 / 2):-int(10 / 2):10]
	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	plt.ylim([-0.2, 0.2])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('feature value in a.u.', fontproperties=font)
	plt.title('10 trials sub-average SSD; interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, '500_900_SSD_10_subs'), dpi=200)
	plt.close('all')


	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx[:8], data_k3_combined[:8])
	ssd_filt_intrplt_filt_500_900_kx = np.tensordot(ssd_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ssd_filt_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(20) / 20, axis=1)[:,int(20 / 2):-int(20 / 2):20]
	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	plt.ylim([-0.2, 0.2])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('feature value in a.u.', fontproperties=font)
	plt.title('20 trials sub-average SSD; interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, '500_900_SSD_20_subs'), dpi=200)
	plt.close('all')


	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx[:8], data_k3_combined[:8])
	ssd_filt_intrplt_filt_500_900_kx = np.tensordot(ssd_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ssd_filt_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(500) / 500, axis=1)[:,int(500 / 2):-int(500 / 2):500]
	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	plt.fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	plt.ylim([-0.2, 0.2])
	#-# plt.xscale('log')
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('feature value in a.u.', fontproperties=font)
	plt.title('500 trials sub-average SSD; interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, '500_900_SSD_500_subs'), dpi=200)
	plt.close('all')


	### recurrence_plot_variations_of_basic_plots
	from pyts.image import RecurrencePlot

	rp = RecurrencePlot()
	# rp.fit_transform(np.swapaxes(x_dat, 0, 1))
	ticks = np.arange(0, 1100, 100)
	ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
	plt.imshow(np.squeeze(rp.fit_transform((data_k3_combined[0,triggers_k3_combined[3]-497:triggers_k3_combined[3]+503]-data_k3_combined[5,triggers_k3_combined[3]-497:triggers_k3_combined[3]+503]).reshape(1, -1)), axis=0))
	plt.colorbar()
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.yticks(ticks=ticks.T, labels=ticklabels)
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('time relative to stimulus in ms', fontproperties=font)
	plt.title('FZ-CP5; stimulus-artifact at 0ms, hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, 'rp_hfSEP_without_preprocessing'), dpi=200)
	plt.close('all')


	ticks = np.arange(0, 1100, 100)
	ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
	plt.imshow(np.squeeze(rp.fit_transform((intrplt_filt_under_100_kx[0,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]-intrplt_filt_under_100_kx[5,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
	plt.colorbar()
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.yticks(ticks=ticks.T, labels=ticklabels)
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('time relative to stimulus in ms', fontproperties=font)
	plt.title('FZ-CP5; data interpolated and IIR-filtered [f <= 100Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, 'rp_hfSEP_intrplt_filt_under_100'), dpi=200)
	plt.close('all')


	ticks = np.arange(0, 1100, 100)
	ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
	plt.imshow(np.squeeze(rp.fit_transform((intrplt_filt_over_100_kx[0,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]-intrplt_filt_over_100_kx[5,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
	plt.colorbar()
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.yticks(ticks=ticks.T, labels=ticklabels)
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('time relative to stimulus in ms', fontproperties=font)
	plt.title('FZ-CP5; data interpolated and IIR-filtered [f >= 100Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, 'rp_hfSEP_intrplt_filt_over_100'), dpi=200)
	plt.close('all')


	ticks = np.arange(0, 1100, 100)
	ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
	plt.imshow(np.squeeze(rp.fit_transform((intrplt_filt_over_400_kx[0,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]-intrplt_filt_over_400_kx[5,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
	plt.colorbar()
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.yticks(ticks=ticks.T, labels=ticklabels)
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('time relative to stimulus in ms', fontproperties=font)
	plt.title('FZ-CP5; data interpolated and IIR-filtered [f >= 400Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, 'rp_hfSEP_intrplt_filt_over_400'), dpi=200)
	plt.close('all')


	ticks = np.arange(0, 1100, 100)
	ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
	plt.imshow(np.squeeze(rp.fit_transform((intrplt_filt_500_900_kx[0,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]-intrplt_filt_500_900_kx[5,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
	plt.colorbar()
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.yticks(ticks=ticks.T, labels=ticklabels)
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('time relative to stimulus in ms', fontproperties=font)
	plt.title('FZ-CP5; data interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, 'rp_hfSEP_intrplt_filt_500_900'), dpi=200)
	plt.close('all')


	ticks = np.arange(0, 1100, 100)
	ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
	plt.imshow(np.squeeze(rp.fit_transform((csp_filt_epoched_intrplt_filt_500_900_kx[triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
	plt.colorbar()
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.yticks(ticks=ticks.T, labels=ticklabels)
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('time relative to stimulus in ms', fontproperties=font)
	plt.title('CSP-filtered data; data interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, 'rp_hfSEP_intrplt_filt_500_900_CSP'), dpi=200)
	plt.close('all')


	ticks = np.arange(0, 1100, 100)
	ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
	plt.imshow(np.squeeze(rp.fit_transform((ccar_filt_epoched_intrplt_filt_500_900_kx[triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
	plt.colorbar()
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.yticks(ticks=ticks.T, labels=ticklabels)
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('time relative to stimulus in ms', fontproperties=font)
	plt.title('CCAr filtered data; data interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, 'rp_hfSEP_intrplt_filt_500_900_CCAr'), dpi=200)
	plt.close('all')


	ticks = np.arange(0, 1100, 100)
	ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
	plt.imshow(np.squeeze(rp.fit_transform((ssd_filt_intrplt_filt_500_900_kx[triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
	plt.colorbar()
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.yticks(ticks=ticks.T, labels=ticklabels)
	plt.xlabel('time relative to stimulus in ms', fontproperties=font)
	plt.ylabel('time relative to stimulus in ms', fontproperties=font)
	plt.title('SSD-filtered data; data interpolated and IIR-filtered [500Hz <= f <= 900Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	plt.tick_params(labelsize=6)
	plt.legend(fontsize=6, loc=2)
	plt.savefig(path % (subject_id, 'rp_hfSEP_intrplt_filt_500_900_SSD'), dpi=200)
	plt.close('all')