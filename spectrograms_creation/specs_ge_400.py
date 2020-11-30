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
version = 'ge400f'
meg_srate = 10000                                                                                                                                                                    

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

intrplt_filt_over_400_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_400_kx.npy' % subject_id, allow_pickle=True)
  
one_sec_win = [0, 10000]                                                                                                                                                             
one_sec_samples_of_ccar_filt_k003 = meet.epochEEG(intrplt_filt_over_400_kx[0] - intrplt_filt_over_400_kx[5], np.arange(0, 1300000, 10000), one_sec_win)

spectograms = list()  
f_range = None  
t_range = None  
for idx, sec in enumerate(one_sec_samples_of_ccar_filt_k003.T):  
  coors, s = meet.tf.gft(sec, axis=0, sampling=custom_sampling_meg)  
  f, t, tf_mean_p_interp = meet.tf.interpolate_gft(coors, s, IM_shape=(len(sec) // 2, len(sec)), data_len=len(sec), kindf='nearest', kindt='linear')   
  #plt.pcolormesh(f[:2501], t[:1001], np.abs(tf_mean_p_interp[:1000, :2500]))  
  #plt.colorbar()  
  tmp = list() 
  tmp0 = np.abs(tf_mean_p_interp[:1500, :2500]) 
  tmp1 = np.abs(tf_mean_p_interp[:1500, 2500:5000]) 
  tmp2 = np.abs(tf_mean_p_interp[:1500, 5000:7500]) 
  tmp3 = np.abs(tf_mean_p_interp[:1500, 7500:]) 
  tmp.append(np.asarray([tmp0[i,:] / (np.concatenate((tmp0[i,:1000], tmp0[i,1600:]), axis=-1).mean()) for i in range(tmp0.shape[0])]))  
  tmp.append(np.asarray([tmp1[i,:] / (np.concatenate((tmp1[i,:1000], tmp1[i,1600:]), axis=-1).mean()) for i in range(tmp1.shape[0])]))  
  tmp.append(np.asarray([tmp2[i,:] / (np.concatenate((tmp2[i,:1000], tmp2[i,1600:]), axis=-1).mean()) for i in range(tmp2.shape[0])]))  
  tmp.append(np.asarray([tmp3[i,:] / (np.concatenate((tmp3[i,:1000], tmp3[i,1600:]), axis=-1).mean()) for i in range(tmp3.shape[0])]))  
  tmp = np.asarray(tmp)  
  spectograms.append(tmp.mean(0))  
  if idx == 0:  
    f_range = f[:1501]  
    t_range = t[:2501] 

spectrograms = np.asarray(spectograms)  

## np.concatenate((mean_spectrogram[i,:1000], mean_spectrogram[i,1600:]), axis=-1).mean()

mean_spectrogram = spectrograms.mean(0)
plt.pcolormesh(t_range, f_range, 20 * np.log10(mean_spectrogram), cmap=plt.cm.plasma, vmin=1, vmax=3)                                                                                                       
bar = plt.colorbar()                                                                                                                                                                
bar.set_label('amplitude', fontsize=20)
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
extension_avg = '7_s%dspec%s500avg'
path = '/home/christoph/Desktop/Thesis_Plots/Final_Versions/' + extension_avg
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig(path % (subject_id, version), dpi=100)
plt.close('all')
np.save(('/home/christoph/Desktop/Thesis_Plots/Final_Versions/npy_objs/specs/' + extension_avg) % (subject_id, version), mean_spectrogram)
np.save(('/home/christoph/Desktop/Thesis_Plots/Final_Versions/npy_objs/specs/' + extension_avg + 'f_range') % (subject_id, version), f_range)
np.save(('/home/christoph/Desktop/Thesis_Plots/Final_Versions/npy_objs/specs/' + extension_avg + 't_range') % (subject_id, version), t_range)


single_trial_spec = spectrograms[20]
plt.pcolormesh(t_range, f_range, 20 * np.log10(single_trial_spec), cmap=plt.cm.plasma, vmin=1, vmax=3)
bar = plt.colorbar()                                                                                                                                                                
bar.set_label('amplitude', fontsize=20)
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
extension_st = '6_s%dspec%ssingletrial'
path = '/home/christoph/Desktop/Thesis_Plots/Final_Versions/' + extension_st
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig(path % (subject_id, version), dpi=100)
plt.close('all')
np.save(('/home/christoph/Desktop/Thesis_Plots/Final_Versions/npy_objs/specs/' + extension_st) % (subject_id, version), single_trial_spec)