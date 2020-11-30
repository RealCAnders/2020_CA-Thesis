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


## Set figure size after creating it prior to saving:
fig, axes = plt.subplots(4, 2, sharex=True)
#-#for i in range(5):
#-#	axes[i, col_id].plot()...
#-#	t = plt.text(0.15, 0.2, 'TEXT N = ....', horizontal_alignmetn=center, verticalalgi=center, transform=axes[row, col_id].transAxes, fontproperties=font)

font = FontProperties()  
font.set_family('serif')  
font.set_name('Times New Roman')  
font.set_size(10)        
ticks = np.arange(-500, 600, 100)
ticklabels = ['%d' % tick for tick in (ticks / 10)]

for subject_id in [1, 2]:
	data_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/kx_data_combined.npy' % subject_id, allow_pickle=True)                                                                  
	triggers_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined.npy' % subject_id, allow_pickle=True)
	intrplt_filt_500_900_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy' % subject_id, allow_pickle=True)

	# basic plot
	i = 0
	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
	csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
	hfsep_around_artifact = meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500])
	std_basic_prep = np.std(hfsep_around_artifact, axis=1)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact.mean(1) - hfsep_around_artifact.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	axes[i, subject_id - 1].fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact[:,3], label='single-trial example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	#-# plt.xscale('log')
	if subject_id == 1:
		axes[i, subject_id - 1].set_ylabel('feature value a.u.', fontproperties=font)
	else:
		axes[i, subject_id - 1].set_xticklabels([])
		axes[i, subject_id - 1].set_yticklabels([])

	axes[i, subject_id - 1].set_ylim([-3, 3])
	# axes[i, subject_id - 1].title('FZ-CP5; artifact at 0ms, wideband signal, hfSEP in [15ms - 30ms]', fontproperties=font)
	t = plt.text(0.11, 0.18, (('K00%d \n' % subject_id) + 'single-trial'), horizontalalignment='center', verticalalignment='center', transform=axes[i, subject_id - 1].transAxes, fontproperties=font)
	t.set_bbox(dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'))
	axes[i, subject_id - 1].tick_params(labelsize=10)
	##axes[i, subject_id - 1].legend(fontsize=10, loc=2)

	### Plots per spectral filter modality
	i = 1
	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
	csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(10) / 10, axis=1)[:,int(10 / 2):-int(10 / 2):10]
	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	axes[i, subject_id - 1].fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
	## axes[i, subject_id - 1].xticks(ticks=ticks, labels=ticklabels)
	## axes[i, subject_id - 1].xlim([-500, 500])
	#-# plt.xscale('log')
	if subject_id == 1:
		axes[i, subject_id - 1].set_ylabel('feature value a.u.', fontproperties=font)
	else:
		axes[i, subject_id - 1].set_xticklabels([])
		axes[i, subject_id - 1].set_yticklabels([])

	axes[i, subject_id - 1].set_ylim([-1, 1])
	# axes[i, subject_id - 1].title('FZ-CP5; data interpolated and IIR-filtered [f <= 100Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	t = plt.text(0.18, 0.18, (('K00%d \n' % subject_id) + 'sub-average w. 10'), horizontalalignment='center', verticalalignment='center', transform=axes[i, subject_id - 1].transAxes, fontproperties=font)
	t.set_bbox(dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'))
	axes[i, subject_id - 1].tick_params(labelsize=10)
	##axes[i, subject_id - 1].legend(fontsize=10, loc=2)

	i = 2
	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
	csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(20) / 20, axis=1)[:,int(20 / 2):-int(20 / 2):20]
	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	axes[i, subject_id - 1].fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
	## axes[i, subject_id - 1].xticks(ticks=ticks, labels=ticklabels)
	## axes[i, subject_id - 1].xlim([-500, 500])
	#-# plt.xscale('log')
	if subject_id == 1:
		axes[i, subject_id - 1].set_ylabel('feature value a.u.', fontproperties=font)
	else:
		axes[i, subject_id - 1].set_xticklabels([])
		axes[i, subject_id - 1].set_yticklabels([])

	axes[i, subject_id - 1].set_ylim([-0.75, 0.75])
	# axes[i, subject_id - 1].title('FZ-CP5; data interpolated and IIR-filtered [f >= 100Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	t = plt.text(0.18, 0.18, (('K00%d \n' % subject_id) + 'sub-average w. 20'), horizontalalignment='center', verticalalignment='center', transform=axes[i, subject_id - 1].transAxes, fontproperties=font)
	t.set_bbox(dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'))
	axes[i, subject_id - 1].tick_params(labelsize=10)
	##axes[i, subject_id - 1].legend(fontsize=10, loc=2)

	i = 3
	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
	csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
	csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(500) / 500, axis=1)[:,int(500 / 2):-int(500 / 2):500]
	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	axes[i, subject_id - 1].fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
	axes[i, subject_id - 1].set_xticks(ticks=ticks)
	axes[i, subject_id - 1].set_xticklabels(labels=ticklabels)
	## axes[i, subject_id - 1].xticks(ticks=ticks, labels=ticklabels)
	## axes[i, subject_id - 1].xlim([-500, 500])
	#-# plt.xscale('log')
	if subject_id == 1:
		axes[i, subject_id - 1].set_ylabel('feature value a.u.', fontproperties=font)
	else:
		axes[i, subject_id - 1].set_yticklabels([])

	axes[i, subject_id - 1].set_xlabel('time relative to stimulus in ms', fontproperties=font)
	axes[i, subject_id - 1].set_ylim([-0.75, 0.75])
	# axes[i, subject_id - 1].title('FZ-CP5; data interpolated and IIR-filtered , hfSEP in [15ms - 30ms]', fontproperties=font)
	t = plt.text(0.19, 0.18, (('K00%d \n' % subject_id) + 'sub-average w. 500'), horizontalalignment='center', verticalalignment='center', transform=axes[i, subject_id - 1].transAxes, fontproperties=font)
	t.set_bbox(dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'))
	axes[i, subject_id - 1].tick_params(labelsize=10)
	##axes[i, subject_id - 1].legend(fontsize=10, loc=2)

lns, lbls = axes[i, subject_id - 1].get_legend_handles_labels()
plt.legend(lns, lbls, fontsize=10, loc='upper center', bbox_to_anchor=(-0.14, 4.95), fancybox=True, shadow=True, ncol=3)

fig.set_size_inches(11.69,8.27)
plt.savefig('/home/christoph/Desktop/Pictures_for_thesis/Final_Combined_Plots/K001_vs_K002_CSP_Trial_Analysis.png', dpi=300)