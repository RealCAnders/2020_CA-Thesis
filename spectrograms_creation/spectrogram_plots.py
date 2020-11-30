# Prepare EPARE NOW ALL THE NICE SPECTROGRAM-PLOTS!!
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

#create a custom sampling scheme for the S transform  
def custom_sampling_meg(N):  
  S_frange = [5, 5000]  
  S_fnum = 30  
  S_Nperperiod = 4  
  wanted_freqs = np.exp(np.linspace(np.log(S_frange[0]), np.log(S_frange[1]), S_fnum))  
  fftfreqs = np.fft.fftfreq(N, d=1./meg_srate)  
  # find the nearest frequency indices  
  y = np.unique([np.argmin((w - fftfreqs)**2) for w in wanted_freqs])  
  x = ((S_Nperperiod*fftfreqs[y]*N/float(meg_srate))//2).astype(int)  
  return x,y 

subject_id = int(sys.argv[1])
version = 'ccar'
meg_srate = 10000                                                                                                                                                                    
subject_id = 3                                                                                                                                                                       

## ### ### ### ###   
## Definition: utility-function / global vars   
## ### ### ### ###   
offset = 1000   
s_rate = 10000   
stim_per_sec = 4   
out_rej_thresh_fz = [0.45, 0.5, 0.225, 0.6, 0.6, 0.4, 0.45, 0.75, 0.45, 2]   
out_rej_thresh_mean = [0.6, 0.415, 0.12, 0.75, 0.3, 0.3, 0.45, 0.45, 0.3, 1.5]                                                                                      
                                 
## ### ### ### ###   
## Data loading   
## ### ### ### ###   
hfSEP_win = [50, 450]   
noise_win = [-500, -100]   
intrplt_win = [-80, 30]                                                                                                                                             

data_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/kx_data_combined.npy' % subject_id, allow_pickle=True)                            
triggers_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined.npy' % subject_id, allow_pickle=True)                    

intrplt_filt_500_900_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy' % subject_id, allow_pickle=True) 

epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % subject_id, allow_pickle=True) 
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % subject_id, allow_pickle=True) 
ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
hfsep_around_artifact = meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500])
std_basic_prep = np.std(hfsep_around_artifact, axis=1)


one_sec_win = [0, 10000]                                                                                                                                                             
one_sec_samples_of_ccar_filt_k003 = meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, np.arange(0, 1300000, 10000), one_sec_win)

spectograms = list() 
f_range = None 
t_range = None 
for idx, sec in enumerate(one_sec_samples_of_ccar_filt_k003.T): 
  coors, s = meet.tf.gft(sec, axis=0, sampling=custom_sampling_meg) 
  f, t, tf_mean_p_interp = meet.tf.interpolate_gft(coors, s, IM_shape=(len(sec) // 2, len(sec)), data_len=len(sec), kindf='nearest', kindt='nearest')  
  #plt.pcolormesh(f[:2501], t[:1001], np.abs(tf_mean_p_interp[:1000, :2500])) 
  #plt.colorbar() 
  tmp = list() 
  tmp.append(np.abs(tf_mean_p_interp[:1500, :2500])) 
  tmp.append(np.abs(tf_mean_p_interp[:1500, 2500:5000])) 
  tmp.append(np.abs(tf_mean_p_interp[:1500, 5000:7500])) 
  tmp.append(np.abs(tf_mean_p_interp[:1500, 7500:])) 
  tmp = np.asarray(tmp) 
  spectograms.append(tmp.mean(0)) 
  if idx == 0: 
    f_range = f[:2501] 
    t_range = t[:1501]

spectrograms = np.asarray(spectograms) 

plt.pcolormesh(f_range, t_range, spectrograms.mean(0), cmap=plt.cm.plasma, vmin=0, vmax=7)                                                                                                          
bar = plt.colorbar()                                                                                                                                                                
bar.set_label('amplitude in a.u.', fontsize=20)
bar.ax.tick_params(labelsize=20)

font = FontProperties()   
font.set_family('serif')   
font.set_name('Times New Roman')   
font.set_size(20)         

ticks = np.arange(0, 2500, 200) 
ticklabels = ['%d' % tick for tick in (ticks / 10) - 100]                                                                                                                               

plt.xticks(ticks=ticks, labels=ticklabels) 
plt.xlim([0, 2400])                                                                                                                                                           
plt.xlabel('time relative to stimulus in ms', fontproperties=font) 
plt.ylabel('frequency (Hz)', fontproperties=font) 
plt.ylim([0, 1400])                                                                                                                                                               
plt.tick_params(labelsize=20)
path = '/home/christoph/Desktop/Thesis_Plots/Final_Versions/s%dspec%s500avg'
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig(path % (subject_id, version), dpi=100)
plt.close('all')

plt.pcolormesh(f_range, t_range, spectrograms[0], cmap=plt.cm.plasma, vmin=0, vmax=7)
bar = plt.colorbar()                                                                                                                                                                
bar.set_label('amplitude in a.u.', fontsize=20)
bar.ax.tick_params(labelsize=20)

font = FontProperties()   
font.set_family('serif')   
font.set_name('Times New Roman')   
font.set_size(20)         

ticks = np.arange(0, 2500, 200) 
ticklabels = ['%d' % tick for tick in (ticks / 10) - 100]                                                                                                                               

plt.xticks(ticks=ticks, labels=ticklabels) 
plt.xlim([0, 2400])                                                                                                                                                           
plt.xlabel('time relative to stimulus in ms', fontproperties=font) 
plt.ylabel('frequency (Hz)', fontproperties=font) 
plt.ylim([0, 1400])                                                                                                                                                               
plt.tick_params(labelsize=20)
path = '/home/christoph/Desktop/Thesis_Plots/Final_Versions/s%dspec%ssingletrial'
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig(path % (subject_id, version), dpi=100)
plt.close('all')

##########################
##########################
##########################

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