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
# ToDo: Move this to the loop!
fig, axes = plt.subplots(3, 2, sharex=True)
#-#for i in range(5):
#-#	axes[i, col_id].plot()...
#-#	t = plt.text(0.15, 0.2, 'TEXT N = ....', horizontal_alignmetn=center, verticalalgi=center, transform=axes[row, col_id].transAxes, fontproperties=font)

font = FontProperties()    
font.set_family('serif')    
font.set_name('Times New Roman')    
font.set_size(10)          

# ToDo: Continue with filling in the limits for the colorbar, continue with moving it all into one loop, utilizing the limits but setting all the other elements there hard
subject_identifiers = [1, 2]
modalities = ['wideband', 'ge400f', 'ccar']
#modalities = ['wideband', 'le100f', 'ge100f', 'ge400f', 'ge500le900f', 'ccar']
limits = [[5, 5], [5, 5], [5, 5]]
#limits = [[10, 10], [10, 10], [10, 10], [3, 3], [3, 3], [3, 3]]
extensions = ['0_s%dspec%ssingletrial','1_s%dspec%s500avg', '6_s%dspec%ssingletrial','7_s%dspec%s500avg', '90_s%dspec%ssingletrial', '91_s%dspec%s500avg']
#extensions = ['0_s%dspec%ssingletrial','1_s%dspec%s500avg', '2_s%dspec%ssingletrial','3_s%dspec%s500avg', '4_s%dspec%ssingletrial','5_s%dspec%s500avg', '6_s%dspec%ssingletrial','7_s%dspec%s500avg', '8_s%dspec%ssingletrial','9_s%dspec%s500avg', '90_s%dspec%ssingletrial', '91_s%dspec%s500avg']

# Testcase:
subject_identifier = 1
for idx, modality in enumerate(modalities):

	print(('/home/christoph/Desktop/Thesis_Plots/Final_Versions/npy_objs/specs/' + extensions[((idx + 1) * 2) - 1] + '.npy') % (subject_identifier, modality))
	mean_spec = np.load(('/home/christoph/Desktop/Thesis_Plots/Final_Versions/npy_objs/specs/' + extensions[((idx + 1) * 2) - 1] + '.npy') % (subject_identifier, modality), allow_pickle=True)
	f_range = np.load(('/home/christoph/Desktop/Thesis_Plots/Final_Versions/npy_objs/specs/' + extensions[((idx + 1) * 2) - 1] + 'f_range' + '.npy') % (subject_identifier, modality), allow_pickle=True)
	t_range = np.load(('/home/christoph/Desktop/Thesis_Plots/Final_Versions/npy_objs/specs/' + extensions[((idx + 1) * 2) - 1] + 't_range' + '.npy') % (subject_identifier, modality), allow_pickle=True)
	single_trial_spec = np.load(('/home/christoph/Desktop/Thesis_Plots/Final_Versions/npy_objs/specs/' + extensions[((idx + 1) * 2) - 2] + '.npy') % (subject_identifier, modality), allow_pickle=True)	

	# single-trial goes left
	if idx < 2:
		img0 = axes[idx, 0].pcolormesh(t_range, f_range, 20 * np.log10(single_trial_spec), cmap=plt.cm.plasma, vmin=1, vmax=3)
	else:
		img0 = axes[idx, 0].pcolormesh(f_range, t_range, 20 * np.log10(single_trial_spec), cmap=plt.cm.plasma, vmin=1, vmax=3)
#-#	bar = plt.colorbar(img0, ax=axes[idx, 0])                                                                                                                                                                 
#-#	bar.set_label('SNNR in dB', fontsize=10) 
#-#	bar.ax.tick_params(labelsize=10) 
	
	# avg goes right
	if idx < 2:
		img = axes[idx, 1].pcolormesh(t_range, f_range, 20 * np.log10(mean_spec), cmap=plt.cm.plasma, vmin=1, vmax=3)                                                                                        
	else:
		img = axes[idx, 1].pcolormesh(f_range, t_range, 20 * np.log10(mean_spec), cmap=plt.cm.plasma, vmin=1, vmax=3)                                                                                                   
	bar = plt.colorbar(img, ax=axes[idx, 1])                                                                                                                                                                 
	bar.set_label('SNNR in dB', fontsize=10) 
	bar.ax.tick_params(labelsize=10) 

axes[0, 0].set_xticklabels([])
axes[0, 1].set_xticklabels([])
axes[1, 0].set_xticklabels([])
axes[1, 1].set_xticklabels([])
axes[0, 1].set_yticklabels([])
axes[1, 1].set_yticklabels([])
axes[2, 1].set_yticklabels([])
axes[0, 0].set_ylabel('frequency [Hz]', fontproperties=font)
axes[1, 0].set_ylabel('frequency [Hz]', fontproperties=font)
axes[2, 0].set_ylabel('frequency [Hz]', fontproperties=font)
axes[2, 0].set_xlabel('time relative to stimulus [ms]', fontproperties=font)
axes[2, 1].set_xlabel('time relative to stimulus [ms]', fontproperties=font)

ticks = np.arange(0, 2500, 200)  
ticklabels = ['%d' % tick for tick in (ticks / 10) - 100]                                                                                                                                
plt.xticks(ticks=ticks, labels=ticklabels)  
plt.xlim([0, 2400])                                                                                                                                                            
plt.ylim([0, 1400])                                                                                                                                                                
plt.tick_params(labelsize=10) 
fig.set_size_inches(14, 9)
plt.savefig('/home/christoph/Desktop/s%dspecs.png' % subject_identifier, dpi=100, bbox_inches='tight')
plt.close('all')


#-#
#-#
#-#	axes[i, subject_id - 1].set_ylim([-3, 3])
#-#	# axes[i, subject_id - 1].title('FZ-CP5; artifact at 0ms, wideband signal, hfSEP in [15ms - 30ms]', fontproperties=font)
#-#	t = plt.text(0.11, 0.18, (('K00%d \n' % subject_id) + 'single-trial'), horizontalalignment='center', verticalalignment='center', transform=axes[i, subject_id - 1].transAxes, fontproperties=font)
#-#	t.set_bbox(dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'))
#-#	axes[i, subject_id - 1].tick_params(labelsize=10)
#-#	##axes[i, subject_id - 1].legend(fontsize=10, loc=2)
#-#
#-#	### Plots per spectral filter modality
#-#	i = 1
#-#	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
#-#	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
#-#	a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
#-#	ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
#-#	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(10) / 10, axis=1)[:,int(10 / 2):-int(10 / 2):10]
#-#	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
#-#	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
#-#	axes[i, subject_id - 1].fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
#-#	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
#-#	## axes[i, subject_id - 1].xticks(ticks=ticks, labels=ticklabels)
#-#	## axes[i, subject_id - 1].xlim([-500, 500])
#-#	#-# plt.xscale('log')
#-#	if subject_id == 1:
#-#		axes[i, subject_id - 1].set_ylabel('feature value a.u.', fontproperties=font)
#-#	else:
#-#		axes[i, subject_id - 1].set_xticklabels([])
#-#		axes[i, subject_id - 1].set_yticklabels([])
#-#
#-#	axes[i, subject_id - 1].set_ylim([-1.25, 1.25])
#-#	# axes[i, subject_id - 1].title('FZ-CP5; data interpolated and IIR-filtered [f <= 100Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
#-#	t = plt.text(0.18, 0.18, (('K00%d \n' % subject_id) + 'sub-average w. 10'), horizontalalignment='center', verticalalignment='center', transform=axes[i, subject_id - 1].transAxes, fontproperties=font)
#-#	t.set_bbox(dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'))
#-#	axes[i, subject_id - 1].tick_params(labelsize=10)
#-#	##axes[i, subject_id - 1].legend(fontsize=10, loc=2)
#-#
#-#	i = 2
#-#	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
#-#	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
#-#	a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
#-#	ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
#-#	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(20) / 20, axis=1)[:,int(20 / 2):-int(20 / 2):20]
#-#	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
#-#	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
#-#	axes[i, subject_id - 1].fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
#-#	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
#-#	## axes[i, subject_id - 1].xticks(ticks=ticks, labels=ticklabels)
#-#	## axes[i, subject_id - 1].xlim([-500, 500])
#-#	#-# plt.xscale('log')
#-#	if subject_id == 1:
#-#		axes[i, subject_id - 1].set_ylabel('feature value a.u.', fontproperties=font)
#-#	else:
#-#		axes[i, subject_id - 1].set_xticklabels([])
#-#		axes[i, subject_id - 1].set_yticklabels([])
#-#
#-#	axes[i, subject_id - 1].set_ylim([-1.25, 1.25])
#-#	# axes[i, subject_id - 1].title('FZ-CP5; data interpolated and IIR-filtered [f >= 100Hz], hfSEP in [15ms - 30ms]', fontproperties=font)
#-#	t = plt.text(0.18, 0.18, (('K00%d \n' % subject_id) + 'sub-average w. 20'), horizontalalignment='center', verticalalignment='center', transform=axes[i, subject_id - 1].transAxes, fontproperties=font)
#-#	t.set_bbox(dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'))
#-#	axes[i, subject_id - 1].tick_params(labelsize=10)
#-#	##axes[i, subject_id - 1].legend(fontsize=10, loc=2)
#-#
#-#	i = 3
#-#	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True)
#-#	epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True)
#-#	a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
#-#	ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
#-#	hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(500) / 500, axis=1)[:,int(500 / 2):-int(500 / 2):500]
#-#	std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
#-#	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten.mean(1) - hfsep_around_artifact_mean_ten.mean(1)[0], label='grand-average', color='black', linewidth=1.25)
#-#	axes[i, subject_id - 1].fill_between(np.arange(-500, 500), -std_basic_prep, std_basic_prep, color='gray', label='+/- STD across single-trials', alpha=0.6)
#-#	axes[i, subject_id - 1].plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], label='sub-average example', color='red', linewidth=0.75)
#-#	axes[i, subject_id - 1].set_xticks(ticks=ticks)
#-#	axes[i, subject_id - 1].set_xticklabels(labels=ticklabels)
#-#	## axes[i, subject_id - 1].xticks(ticks=ticks, labels=ticklabels)
#-#	## axes[i, subject_id - 1].xlim([-500, 500])
#-#	#-# plt.xscale('log')
#-#	if subject_id == 1:
#-#		axes[i, subject_id - 1].set_ylabel('feature value a.u.', fontproperties=font)
#-#	else:
#-#		axes[i, subject_id - 1].set_yticklabels([])
#-#
#-#	axes[i, subject_id - 1].set_xlabel('time relative to stimulus in ms', fontproperties=font)
#-#	axes[i, subject_id - 1].set_ylim([-1.25, 1.25])
#-#	# axes[i, subject_id - 1].title('FZ-CP5; data interpolated and IIR-filtered , hfSEP in [15ms - 30ms]', fontproperties=font)
#-#	t = plt.text(0.19, 0.18, (('K00%d \n' % subject_id) + 'sub-average w. 500'), horizontalalignment='center', verticalalignment='center', transform=axes[i, subject_id - 1].transAxes, fontproperties=font)
#-#	t.set_bbox(dict(facecolor='lightgray', alpha=0.5, edgecolor='gray'))
#-#	axes[i, subject_id - 1].tick_params(labelsize=10)
#-#	##axes[i, subject_id - 1].legend(fontsize=10, loc=2)
#-#
#-#lns, lbls = axes[i, subject_id - 1].get_legend_handles_labels()
#-#plt.legend(lns, lbls, fontsize=10, loc='upper center', bbox_to_anchor=(-0.14, 4.95), fancybox=True, shadow=True, ncol=3)
#-#
#-#fig.set_size_inches(11.69,8.27)
#-#plt.savefig('/home/christoph/Desktop/Pictures_for_thesis/Final_Combined_Plots/SpecK002Part1.png', dpi=100)
#-#
#-#
#-##################
#-##################
#-##################
#-#
#-#plt.pcolormesh(f_range, t_range, mean_spectrogram_divided_by_mean, cmap=plt.cm.plasma, vmin=0, vmax=50)                                                                                                          
#-#bar = plt.colorbar()                                                                                                                                                                
#-#bar.set_label('amplitude', fontsize=20)
#-#bar.ax.tick_params(labelsize=20)
#-#
#-#font = FontProperties()   
#-#font.set_family('serif')   
#-#font.set_name('Times New Roman')   
#-#font.set_size(20)         
#-#
#-#ticks = np.arange(0, 2500, 200) 
#-#ticklabels = ['%d' % tick for tick in (ticks / 10) - 100]                                                                                                                               
#-#
#-#plt.xticks(ticks=ticks, labels=ticklabels) 
#-#plt.xlim([0, 2400])                                                                                                                                                           
#-#plt.xlabel('time relative to stimulus in ms', fontproperties=font) 
#-#plt.ylabel('frequency (Hz)', fontproperties=font) 
#-#plt.ylim([0, 1400])                                                                                                                                                               
#-#plt.tick_params(labelsize=20)
#-#extension_avg = '1_s%dspec%s500avg'
#-#path = '/home/christoph/Desktop/Thesis_Plots/Final_Versions/' + extension_avg
#-#fig = plt.gcf()
#-#fig.set_size_inches(18.5, 10.5)
#-#fig.savefig(path % (subject_id, version), dpi=100)
#-#plt.close('all')
#-#
#-#
#-#single_trial_spec = spectrograms[0]
#-#st_spec_div_mean = np.asarray([single_trial_spec[i,:] - single_trial_spec[i,:].mean() for i in range(single_trial_spec.shape[0])])
#-#plt.pcolormesh(f_range, t_range, st_spec_div_mean, cmap=plt.cm.plasma, vmin=0, vmax=30)
#-#bar = plt.colorbar()                                                                                                                                                                
#-#bar.set_label('amplitude', fontsize=20)
#-#bar.ax.tick_params(labelsize=20)
#-#
#-#font = FontProperties()   
#-#font.set_family('serif')   
#-#font.set_name('Times New Roman')   
#-#font.set_size(20)         
#-#
#-#ticks = np.arange(0, 2500, 200) 
#-#ticklabels = ['%d' % tick for tick in (ticks / 10) - 100]                                                                                                                               
#-#
#-#plt.xticks(ticks=ticks, labels=ticklabels) 
#-#plt.xlim([0, 2400])                                                                                                                                                           
#-#plt.xlabel('time relative to stimulus in ms', fontproperties=font) 
#-#plt.ylabel('frequency (Hz)', fontproperties=font) 
#-#plt.ylim([0, 1400])                                                                                                                                                               
#-#plt.tick_params(labelsize=20)
#-#extension_st = '0_s%dspec%ssingletrial'
#-#path = '/home/christoph/Desktop/Thesis_Plots/Final_Versions/' + extension_st
#-#fig = plt.gcf()
#-#fig.set_size_inches(18.5, 10.5)
#-#fig.savefig(path % (subject_id, version), dpi=100)
#-#plt.close('all')