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
version = 'le100f'
meg_srate = 10000                                                                                                                                                                    

## ### ### ### ###   
## Definition: utility-function / global vars   
## ### ### ### ###   
offset = 1000   
s_rate = 10000   
stim_per_sec = 4   
out_rej_thresh_fz = [0.45, 0.5, 0.225, 0.6, 0.6, 0.4, 0.45, 0.75, 0.45, 2]   
out_rej_thresh_mean = [0.6, 0.415, 0.12, 0.75, 0.3, 0.3, 0.45, 0.45, 0.3, 1.5]                                                                                      
                                 

# different plots!
# In [87]: plt.imshow(x_transformed_binary[0], cmap=plt.cm.binary)                                                                                                            


## ### ### ### ###   
## Data loading   
## ### ### ### ###   
hfSEP_win = [50, 450]   
noise_win = [-500, -100]   
intrplt_win = [-80, 30]                                                                                                                                             

triggers_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined.npy' % subject_id, allow_pickle=True)                    

intrplt_filt_under_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_under_100_kx.npy' % subject_id, allow_pickle=True)

from pyts.image import RecurrencePlot                                                                                                                                                                     
rp = RecurrencePlot()                                                                                                                                                                                     
hfsep_around_artifact = meet.epochEEG(intrplt_filt_under_100_kx, triggers_k3_combined, [-497, 503])                                                                                                              
rp_mods = hfsep_around_artifact[0] - hfsep_around_artifact[5]                                                                                                                                             
rps_hfseps = np.asarray([rp.fit_transform(rp_mods[:,i].reshape(1, -1)) for i in range(500)])                                                                                                             
rps_hfseps = rps_hfseps.squeeze()                                                                                                                                                                        
mean_rp = rps_hfseps.mean(0)

font = FontProperties()   
font.set_family('serif')   
font.set_name('Times New Roman')   
font.set_size(20)         

# x_transformed = rp.fit_transform((rps_hfseps[0]).reshape(1, -1)) 
plt.imshow(rps_hfseps[0], cmap=plt.cm.plasma, vmin=0, vmax=10)                                                                                                                                      
plt.yticks(ticks=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], labels=['-50', '-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50'])
plt.xticks(ticks=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], labels=['-50', '-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50'])
plt.xlim([0, 1000]) 
plt.ylim([0, 1000]) 
plt.xlabel('Time in seconds', fontsize=20) 
plt.ylabel('Time in seconds', fontsize=20) 
bar = plt.colorbar()   
bar.set_label('Distance', fontsize=20)  
bar.ax.tick_params(labelsize=20)
plt.tick_params(labelsize=20)
extension_st = 's%d_2_recplot%ssingletrial'
path = '/home/christoph/Desktop/Thesis_Plots/Final_Versions/' + extension_st
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig(path % (subject_id, version), dpi=100)
plt.close('all')
np.save(('/home/christoph/Desktop/Thesis_Plots/Final_Versions/npy_objs/recs/' + extension_st) % (subject_id, version), rps_hfseps[0])

plt.imshow(mean_rp, cmap=plt.cm.plasma, vmin=0, vmax=10)                                                                                                                                      
plt.yticks(ticks=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], labels=['-50', '-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50'])
plt.xticks(ticks=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], labels=['-50', '-40', '-30', '-20', '-10', '0', '10', '20', '30', '40', '50'])
plt.xlim([0, 1000]) 
plt.ylim([0, 1000]) 
plt.xlabel('Time in seconds', fontsize=20) 
plt.ylabel('Time in seconds', fontsize=20) 
bar = plt.colorbar()   
bar.set_label('Distance', fontsize=20)
bar.ax.tick_params(labelsize=20)
plt.tick_params(labelsize=20)
extension_avg = 's%d_3_recplot%s500eravg'
path = '/home/christoph/Desktop/Thesis_Plots/Final_Versions/' + extension_avg
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig(path % (subject_id, version), dpi=100)
plt.close('all')
np.save(('/home/christoph/Desktop/Thesis_Plots/Final_Versions/npy_objs/recs/' + extension_avg) % (subject_id, version), mean_rp)