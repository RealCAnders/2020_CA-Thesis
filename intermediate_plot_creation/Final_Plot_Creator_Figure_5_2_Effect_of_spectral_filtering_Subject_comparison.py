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
fig, axes = plt.subplots(5, 2, sharex=True)
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

	# basic plot
	i = 0
	hfsep_around_artifact = meet.epochEEG(data_k3_combined, triggers_k3_combined, [-497, 503])
	std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1) - (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	axes[i, subject_id - 1].fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), (hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500]) - (hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500]).mean(), label='single-trial example', color='red', linewidth=0.75)
	plt.xticks(ticks=ticks, labels=ticklabels)
	plt.xlim([-500, 500])
	#-# plt.xscale('log')
	if subject_id == 1:
		axes[i, subject_id - 1].set_ylabel('voltage in μV', fontproperties=font)
	else:
		axes[i, subject_id - 1].set_xticklabels([])
		axes[i, subject_id - 1].set_yticklabels([])

	axes[i, subject_id - 1].set_ylim([-50, 50])
	# axes[i, subject_id - 1].title('FZ-CP5; artifact at 0ms, wideband signal, hfSEP in [15ms - 30ms]', fontproperties=font)
	t = plt.text(0.17, 0.2, (('K00%d \n' % subject_id) + r'$wideband-signal$'), horizontalalignment='center', verticalalignment='center', transform=axes[i, subject_id - 1].transAxes, fontproperties=font)
	t.set_bbox(dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'))
	axes[i, subject_id - 1].tick_params(labelsize=10)
	##axes[i, subject_id - 1].legend(fontsize=10, loc=2)

	### Plots per spectral filter modality
	i = 1
	intrplt_filt_under_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_under_100_kx.npy' % subject_id, allow_pickle=True)
	hfsep_around_artifact = meet.epochEEG(intrplt_filt_under_100_kx, triggers_k3_combined, [-500, 500])
	std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1) - (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	axes[i, subject_id - 1].fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500] - (hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500]).mean(), label='single-trial example', color='red', linewidth=0.75)
	## axes[i, subject_id - 1].xticks(ticks=ticks, labels=ticklabels)
	## axes[i, subject_id - 1].xlim([-500, 500])
	#-# plt.xscale('log')
	if subject_id == 1:
		axes[i, subject_id - 1].set_ylabel('voltage in μV', fontproperties=font)
	else:
		axes[i, subject_id - 1].set_xticklabels([])
		axes[i, subject_id - 1].set_yticklabels([])

	axes[i, subject_id - 1].set_ylim([-50, 50])
	# axes[i, subject_id - 1].title('FZ-CP5; data interpolated and IIR-filtered [f <= 100Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	t = plt.text(0.11, 0.2, (('K00%d \n' % subject_id) + r'$[f \leq 100Hz]$'), horizontalalignment='center', verticalalignment='center', transform=axes[i, subject_id - 1].transAxes, fontproperties=font)
	t.set_bbox(dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'))
	axes[i, subject_id - 1].tick_params(labelsize=10)
	##axes[i, subject_id - 1].legend(fontsize=10, loc=2)

	i = 2
	intrplt_filt_over_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_100_kx.npy' % subject_id, allow_pickle=True)
	hfsep_around_artifact = meet.epochEEG(intrplt_filt_over_100_kx, triggers_k3_combined, [-500, 500])
	std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1) - (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	axes[i, subject_id - 1].fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500]  - (hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500]).mean(), label='single-trial example', color='red', linewidth=0.75)
	## axes[i, subject_id - 1].xticks(ticks=ticks, labels=ticklabels)
	## axes[i, subject_id - 1].xlim([-500, 500])
	#-# plt.xscale('log')
	if subject_id == 1:
		axes[i, subject_id - 1].set_ylabel('voltage in μV', fontproperties=font)
	else:
		axes[i, subject_id - 1].set_xticklabels([])
		axes[i, subject_id - 1].set_yticklabels([])

	axes[i, subject_id - 1].set_ylim([-10, 10])
	# axes[i, subject_id - 1].title('FZ-CP5; data interpolated and IIR-filtered [f >= 100Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
	t = plt.text(0.11, 0.2, (('K00%d \n' % subject_id) + r'$[f \geq 100Hz]$'), horizontalalignment='center', verticalalignment='center', transform=axes[i, subject_id - 1].transAxes, fontproperties=font)
	t.set_bbox(dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'))
	axes[i, subject_id - 1].tick_params(labelsize=10)
	##axes[i, subject_id - 1].legend(fontsize=10, loc=2)

	i = 3
	intrplt_filt_over_400_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_400_kx.npy' % subject_id, allow_pickle=True)
	hfsep_around_artifact = meet.epochEEG(intrplt_filt_over_400_kx, triggers_k3_combined, [-500, 500])
	std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1) - (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	axes[i, subject_id - 1].fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500] - (hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500]).mean(), label='single-trial example', color='red', linewidth=0.75)
	## axes[i, subject_id - 1].xticks(ticks=ticks, labels=ticklabels)
	## axes[i, subject_id - 1].xlim([-500, 500])
	#-# plt.xscale('log')
	if subject_id == 1:
		axes[i, subject_id - 1].set_ylabel('voltage in μV', fontproperties=font)
	else:
		axes[i, subject_id - 1].set_xticklabels([])
		axes[i, subject_id - 1].set_yticklabels([])

	axes[i, subject_id - 1].set_ylim([-3, 3])
	# axes[i, subject_id - 1].title('FZ-CP5; data interpolated and IIR-filtered [], hfSEP in [15ms - 30ms]', fontproperties=font)
	t = plt.text(0.11, 0.2, (('K00%d \n' % subject_id) + r'$[f \geq 400Hz]$'), horizontalalignment='center', verticalalignment='center', transform=axes[i, subject_id - 1].transAxes, fontproperties=font)
	t.set_bbox(dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'))
	axes[i, subject_id - 1].tick_params(labelsize=10)
	##axes[i, subject_id - 1].legend(fontsize=10, loc=2)

	i = 4
	intrplt_filt_500_900_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy' % subject_id, allow_pickle=True)
	hfsep_around_artifact = meet.epochEEG(intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500])
	std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1) - (hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:]).mean(1)[0], label='grand-average', color='black', linewidth=1.25)
	axes[i, subject_id - 1].fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500] - (hfsep_around_artifact[0,:,500]-hfsep_around_artifact[5,:,500]).mean(), label='single-trial example', color='red', linewidth=0.75)
	axes[i, subject_id - 1].set_xticks(ticks=ticks)
	axes[i, subject_id - 1].set_xticklabels(labels=ticklabels)
	axes[i, subject_id - 1].set_xlim([-500, 500])
	if subject_id == 1:
		axes[i, subject_id - 1].set_ylabel('voltage in μV', fontproperties=font)
	else:
		axes[i, subject_id - 1].set_yticklabels([])

	axes[i, subject_id - 1].set_xlabel('time relative to stimulus in ms', fontproperties=font)
	axes[i, subject_id - 1].set_ylim([-3, 3])
	# axes[i, subject_id - 1].title('FZ-CP5; data interpolated and IIR-filtered , hfSEP in [15ms - 30ms]', fontproperties=font)
	t = plt.text(0.185, 0.2, (('K00%d \n' % subject_id) + r'$[500Hz \leq f \leq 900Hz]$'), horizontalalignment='center', verticalalignment='center', transform=axes[i, subject_id - 1].transAxes, fontproperties=font)
	t.set_bbox(dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'))
	axes[i, subject_id - 1].tick_params(labelsize=10)
	##axes[i, subject_id - 1].legend(fontsize=10, loc=2)

lns, lbls = axes[i, subject_id - 1].get_legend_handles_labels()
plt.legend(lns, lbls, fontsize=10, loc='upper center', bbox_to_anchor=(-0.14, 6.25), fancybox=True, shadow=True, ncol=3)

fig.set_size_inches(11.69,8.27)
plt.savefig('/home/christoph/Desktop/Pictures_for_thesis/Final_Combined_Plots/K001_vs_K002_Spectral_Modalities_Single_Trial_Analysis.png', dpi=300)