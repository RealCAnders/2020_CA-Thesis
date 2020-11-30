import meet 
import numpy as np 
import scipy 
import matplotlib.pyplot as plt 
from numpy import load 

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
#-#for i in range(5):
#-#	axes[i, col_id].plot()...
#-#	t = plt.text(0.85, 0.84, 'TEXT N = ....', horizontal_alignmetn=center, verticalalgi=center, transform=axes[row, col_id].transAxes, fontproperties=font)

font = FontProperties()  
font.set_family('serif')  
font.set_name('times New Roman')  
font.set_size(10)        
ticks = np.arange(-500, 600, 100)
ticklabels = ['%d' % tick for tick in (ticks / 10)]

for subject_id in [1, 2]:

	fig, axes = plt.subplots(4, 3, sharex=True)

	# WIDEBAND
	data_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/kx_data_combined.npy' % subject_id, allow_pickle=True)                            
	triggers_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined.npy' % subject_id, allow_pickle=True)   

	from pyts.image import RecurrencePlot                                                                                                                                                                     
	rp = RecurrencePlot()                                                                                                                                                                                     
	hfsep_around_artifact = meet.epochEEG(data_k3_combined, triggers_k3_combined, [-497, 503])                                                                                                              
	rp_mods = hfsep_around_artifact[0] - hfsep_around_artifact[5]                                                                                                                                             
	rps_hfseps = np.asarray([rp.fit_transform(rp_mods[:,i].reshape(1, -1)) for i in range(500)])                                                                                                             
	rps_hfseps = rps_hfseps.squeeze()                                                                                                                                                                        
	mean_rp = rps_hfseps.mean(0)

	font = FontProperties()   
	font.set_family('serif')   
	font.set_name('times New Roman')   
	font.set_size(20)         

	if subject_id == 1:
		img = axes[0, 0].imshow(rps_hfseps[5], cmap=plt.cm.plasma, vmin=0, vmax=8)                                                                                                                                      
	else:
		img = axes[0, 0].imshow(rps_hfseps[15], cmap=plt.cm.plasma, vmin=0, vmax=8)
	img0 = axes[1, 0].imshow(mean_rp, cmap=plt.cm.plasma, vmin=0, vmax=8)                                                                                                                                      

	# INTRPLT-LE-100
	intrplt_filt_under_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_under_100_kx.npy' % subject_id, allow_pickle=True)

	rp = RecurrencePlot()                                                                                                                                                                                     
	hfsep_around_artifact = meet.epochEEG(intrplt_filt_under_100_kx, triggers_k3_combined, [-497, 503])                                                                                                              
	rp_mods = hfsep_around_artifact[0] - hfsep_around_artifact[5]                                                                                                                                             
	rps_hfseps = np.asarray([rp.fit_transform(rp_mods[:,i].reshape(1, -1)) for i in range(500)])                                                                                                             
	rps_hfseps = rps_hfseps.squeeze()                                                                                                                                                                        
	mean_rp = rps_hfseps.mean(0)

	if subject_id == 1:
		img = axes[0, 1].imshow(rps_hfseps[5], cmap=plt.cm.plasma, vmin=0, vmax=8)                                                                                                                                      
	else:
		img = axes[0, 1].imshow(rps_hfseps[15], cmap=plt.cm.plasma, vmin=0, vmax=8)
	img0 = axes[1, 1].imshow(mean_rp, cmap=plt.cm.plasma, vmin=0, vmax=8)                                                                                                                                      

	# INTRPLT-GE-100
	intrplt_filt_over_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_100_kx.npy' % subject_id, allow_pickle=True)

	rp = RecurrencePlot()                                                                                                                                                                                     
	hfsep_around_artifact = meet.epochEEG(intrplt_filt_over_100_kx, triggers_k3_combined, [-497, 503])                                                                                                              
	rp_mods = hfsep_around_artifact[0] - hfsep_around_artifact[5]                                                                                                                                             
	rps_hfseps = np.asarray([rp.fit_transform(rp_mods[:,i].reshape(1, -1)) for i in range(500)])                                                                                                             
	rps_hfseps = rps_hfseps.squeeze()                                                                                                                                                                        
	mean_rp = rps_hfseps.mean(0)

	if subject_id == 1:
		img = axes[0, 2].imshow(rps_hfseps[5], cmap=plt.cm.plasma, vmin=0, vmax=8)                                                                                                                                      
	else:
		img = axes[0, 2].imshow(rps_hfseps[15], cmap=plt.cm.plasma, vmin=0, vmax=8)
	bar = plt.colorbar(img, ax=axes[0, 2], ticks=[0, 4, 8])   
	bar.set_label('distance', fontsize=10)  
	bar.ax.tick_params(labelsize=10)
	bar.ax.set_yticklabels(['0', '4', '8'])
	
	img0 = axes[1, 2].imshow(mean_rp, cmap=plt.cm.plasma, vmin=0, vmax=8)                                                                                                                                      
	bar = plt.colorbar(img0, ax=axes[1, 2], ticks=[0, 4, 8])   
	bar.set_label('distance', fontsize=10)
	bar.ax.tick_params(labelsize=10)
	bar.ax.set_yticklabels(['0', '4', '8'])

	# INTRPLT-GE-400
	intrplt_filt_over_400_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_400_kx.npy' % subject_id, allow_pickle=True)

	rp = RecurrencePlot()                                                                                                                                                                                     
	hfsep_around_artifact = meet.epochEEG(intrplt_filt_over_400_kx, triggers_k3_combined, [-497, 503])                                                                                                              
	rp_mods = hfsep_around_artifact[0] - hfsep_around_artifact[5]                                                                                                                                             
	rps_hfseps = np.asarray([rp.fit_transform(rp_mods[:,i].reshape(1, -1)) for i in range(500)])                                                                                                             
	rps_hfseps = rps_hfseps.squeeze()                                                                                                                                                                        
	mean_rp = rps_hfseps.mean(0)

	if subject_id == 1:
		img = axes[2, 0].imshow(rps_hfseps[5], cmap=plt.cm.plasma, vmin=0, vmax=2)                                                                                                                                      
	else:
		img = axes[2, 0].imshow(rps_hfseps[15], cmap=plt.cm.plasma, vmin=0, vmax=2)                                                                                                                                      
	img0 = axes[3, 0].imshow(mean_rp, cmap=plt.cm.plasma, vmin=0, vmax=2)                                                                                                                                      

	# INTRPLT-500-f-900
	intrplt_filt_500_900_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy' % subject_id, allow_pickle=True)

	rp = RecurrencePlot()                                                                                                                                                                                     
	hfsep_around_artifact = meet.epochEEG(intrplt_filt_500_900_kx, triggers_k3_combined, [-497, 503])                                                                                                              
	rp_mods = hfsep_around_artifact[0] - hfsep_around_artifact[5]                                                                                                                                             
	rps_hfseps = np.asarray([rp.fit_transform(rp_mods[:,i].reshape(1, -1)) for i in range(500)])                                                                                                             
	rps_hfseps = rps_hfseps.squeeze()                                                                                                                                                                        
	mean_rp = rps_hfseps.mean(0)

	if subject_id == 1:
		img = axes[2, 1].imshow(rps_hfseps[5], cmap=plt.cm.plasma, vmin=0, vmax=2)                                                                                                                                      
	else:
		img = axes[2, 1].imshow(rps_hfseps[15], cmap=plt.cm.plasma, vmin=0, vmax=2)
	img0 = axes[3, 1].imshow(mean_rp, cmap=plt.cm.plasma, vmin=0, vmax=2)                                                                                                                                      

	# INTRPLT-500-f-900-CCAr
	intrplt_filt_500_900_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy' % subject_id, allow_pickle=True)
	epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True) 
	a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
	ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))

	rp = RecurrencePlot()                                                                                                                                                                                     
	rp_mods = meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-497, 503])                                                                                                              
	rps_hfseps = np.asarray([rp.fit_transform(rp_mods[:,i].reshape(1, -1)) for i in range(500)])                                                                                                             
	rps_hfseps = rps_hfseps.squeeze()                                                                                                                                                                        
	mean_rp = rps_hfseps.mean(0)

	if subject_id == 1:
		img = axes[2, 2].imshow(rps_hfseps[5], cmap=plt.cm.plasma, vmin=0, vmax=2)                                                                                                                                      
	else:
		img = axes[2, 2].imshow(rps_hfseps[15], cmap=plt.cm.plasma, vmin=0, vmax=2)
	bar = plt.colorbar(img, ax=axes[2, 2], ticks=[0, 1, 2])   
	bar.set_label('distance', fontsize=10)  
	bar.ax.tick_params(labelsize=10)
	bar.ax.set_yticklabels(['0', '1', '2'])
	
	img0 = axes[3, 2].imshow(mean_rp, cmap=plt.cm.plasma, vmin=0, vmax=2)                                                                                                                                      
	bar = plt.colorbar(img0, ax=axes[3, 2], ticks=[0, 1, 2])   
	bar.set_label('distance', fontsize=10)
	bar.ax.tick_params(labelsize=10)
	bar.ax.set_yticklabels(['0', '1', '2'])

	for a in [0, 1, 2]:
		for b in range(4):
			axes[b, a].set_yticks(ticks=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
			axes[b, a].set_xticks(ticks=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
			axes[b, a].set_xlim([0, 1000]) 
			axes[b, a].set_ylim([0, 1000]) 

	axes[0, 0].set_yticklabels(labels=['-50', '-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50'], fontsize=10)
	axes[1, 0].set_yticklabels(labels=['-50', '-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50'], fontsize=10)
	axes[2, 0].set_yticklabels(labels=['-50', '-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50'], fontsize=10)
	axes[3, 0].set_yticklabels(labels=['-50', '-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50'], fontsize=10)
	axes[0, 1].set_yticklabels(labels=[], fontsize=10)
	axes[1, 1].set_yticklabels(labels=[], fontsize=10)
	axes[2, 1].set_yticklabels(labels=[], fontsize=10)
	axes[3, 1].set_yticklabels(labels=[], fontsize=10)
	axes[0, 2].set_yticklabels(labels=[], fontsize=10)
	axes[1, 2].set_yticklabels(labels=[], fontsize=10)
	axes[2, 2].set_yticklabels(labels=[], fontsize=10)
	axes[3, 2].set_yticklabels(labels=[], fontsize=10)
	axes[3, 0].set_xticklabels(labels=['-50', '-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50'], fontsize=10, rotation=90)
	axes[3, 1].set_xticklabels(labels=['-50', '-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50'], fontsize=10, rotation=90)
	axes[3, 2].set_xticklabels(labels=['-50', '-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50'], fontsize=10, rotation=90)
	axes[0, 0].set_ylabel('time in ms', fontsize=10)
	axes[1, 0].set_ylabel('time in ms', fontsize=10)
	axes[2, 0].set_ylabel('time in ms', fontsize=10)
	axes[3, 0].set_ylabel('time in ms', fontsize=10)
	axes[3, 0].set_xlabel('time in ms', fontsize=10)
	axes[3, 1].set_xlabel('time in ms', fontsize=10)
	axes[3, 2].set_xlabel('time in ms', fontsize=10)
	
	fig.set_size_inches(14, 12)
	plt.savefig('/home/christoph/Desktop/Thesis_Plots/Final_Versions/rec_plot_last_minute/recfors%dcombinedinraster.png' % subject_id, dpi=100, bbox_inches='tight')