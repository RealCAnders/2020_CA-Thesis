import sys
import meet  
import numpy as np  
import matplotlib.pyplot as plt  
import scipy
from scipy import signal  
from scipy.fft import fftshift  
from scipy.ndimage import convolve1d, convolve  
from numpy import save
from meet.spatfilt import CSP

### ### ### ### ###
### Definition: utility-function / global vars
### ### ### ### ###
out_rej_thresh_fz = [0.45, 0.5, 0.225, 0.6, 0.6, 0.4, 0.45, 0.75, 0.45, 2]
out_rej_thresh_mean = [0.6, 0.415, 0.12, 0.75, 0.3, 0.3, 0.45, 0.45, 0.3, 1.5]

def identify_triggers(trigger_signal, estimated_trigger_distance, indicator_value):  
  
    # 1st version: define the timestamp when the signal is at zero again as "start of trigger"  
    triggers = [0]  
    ttl_found = False  
    ttl_samples_ctr = 0  
      
    for idx, data_point in enumerate(trigger_signal):  
        if triggers[-1] + int(0.9 * estimated_trigger_distance) <= idx and trigger_signal[idx] == indicator_value:  
            ttl_found = True  
            ttl_samples_ctr = ttl_samples_ctr + 1  
        else:  
            ttl_found = False  
        if ttl_samples_ctr > 0 and not ttl_found:  
            triggers.append(idx + 40) # -1 as to change of index for old position; -41 as to offset-correciton  
            ttl_samples_ctr = 0  
          
    return triggers[1:] 

def normalize_min_max(to_be_normalized): 
    return (to_be_normalized - np.min(to_be_normalized)) / (np.max(to_be_normalized) - np.min(to_be_normalized)) 

def normalize_z(to_be_normalized): 
    return (to_be_normalized - np.mean(to_be_normalized)) / (np.std(to_be_normalized)) 

def reject_outlier_seconds_using_lsd(dat, chan=None, measure='ED', thresh=0.13, plot_eds=False, srate=10000, identifier=-1): 
    ### ToDo: Document and keep in mind the situation of max_until_... 
    lsd_baseline = [] 
    euclidean_distances = [] 
    num_secs = int(len(dat[0]) / srate)
 
    if chan is None: 
        pxx_per_sec = np.asarray([scipy.signal.welch(dat[:8, srate * i : srate * (i + 1)], fs=srate)[1] for i in range(num_secs)]) 
        lsd_baseline = np.median(pxx_per_sec, axis=0).mean(0) 
        euclidean_distances = np.sqrt(((lsd_baseline - pxx_per_sec.mean(1))**2).sum(1)) 
    else: 
        pxx_per_sec = np.asarray([scipy.signal.welch(dat[chan, srate * i : srate * (i + 1)], fs=srate)[1] for i in range(num_secs)])  
        lsd_baseline = np.median(pxx_per_sec, axis=0).mean(0) 
        euclidean_distances = np.sqrt(((lsd_baseline - pxx_per_sec)**2).sum(1)) 
     
    if plot_eds: 
        import matplotlib.pyplot as plt
        plt.plot(euclidean_distances, color='blue', label='ED in channel FZ') 
        plt.plot(np.ones(len(euclidean_distances)) * (0.25 * 3), color='yellow', linestyle='-.', label='Threshold at %f' % (0.25 * 3)) 
        plt.plot(np.ones(len(euclidean_distances)) * (0.2 * 3), color='black', linestyle=':', label='Threshold at %f' % (0.2 * 3)) 
        plt.plot(np.ones(len(euclidean_distances)) * (0.15 * 3), color='yellow', linestyle='--', label='Threshold at %f' % (0.15 * 3)) 
        plt.plot(np.ones(len(euclidean_distances)) * (0.1 * 3), color='black', linestyle='--', label='Threshold at %f' % (0.1 * 3)) 
        plt.title('ED-Plot to visually derive the threshold for outlier rejection of K00%d' % identifier) 
        plt.xlabel('Seconds in combined data of K00%d; intrplt. stim. 1, 2, 3 comb.; Zoomed in' % identifier) 
        plt.ylabel('Euclidean Distance') 
        plt.ylim([0.0, 3.0]) 
        plt.legend() 
        plt.savefig('/media/christoph/Volume/Masterthesis/Presentations/Zwischenpräsentation_18_09/k00%d_ED_Plot_FZ_Zoomed_In' % identifier) 
        plt.clf()
     
    ed_mean = euclidean_distances.mean() 
    ed_std = euclidean_distances.std() 
    
    lsds_kept_within_ed_threshold = 0 
    lsds_kept_one_sd_over_mean = 0 
    lsds_kept_two_sd_over_mean = 0 

    if chan is None: 
       lsds_kept_within_ed_threshold = np.arange(0, len(euclidean_distances), 1)[euclidean_distances <= out_rej_thresh_mean[identifier]] 
       lsds_kept_one_sd_over_mean = np.arange(0, len(euclidean_distances), 1)[(euclidean_distances - ed_mean - ed_std) <= 0] 
       lsds_kept_two_sd_over_mean = np.arange(0, len(euclidean_distances), 1)[(euclidean_distances - ed_mean - (2*ed_std) <= 0)] 
    else:
       lsds_kept_within_ed_threshold = np.arange(0, len(euclidean_distances), 1)[euclidean_distances <= out_rej_thresh_fz[identifier]] 
       lsds_kept_one_sd_over_mean = np.arange(0, len(euclidean_distances), 1)[(euclidean_distances - ed_mean - ed_std) <= 0] 
       lsds_kept_two_sd_over_mean = np.arange(0, len(euclidean_distances), 1)[(euclidean_distances - ed_mean - (2*ed_std) <= 0)]
     
    secs_to_keep = [] 
    if measure is 'ED': 
        secs_to_keep = lsds_kept_within_ed_threshold 
    elif measure is 'ED1SD': 
        secs_to_keep = lsds_kept_one_sd_over_mean 
    elif measure is 'ED2SD': 
        secs_to_keep = lsds_kept_two_sd_over_mean 
    else: 
        raise ValueError('Unknown measure %s specified; Must be one of: {ED, ED1SD, ED2SD}' % measure) 
         
    conc_kept_data_ids = np.concatenate([np.full(srate, True) if i in secs_to_keep else np.full(srate, False) for i in range(num_secs)]) 
    return dat[:,conc_kept_data_ids]

def eigh(cov1, cov2):
    rank = np.linalg.matrix_rank(cov2)
    w, v = np.linalg.eigh(cov2)
    # get whitening matrix
    W = v[:,-rank:]/np.sqrt(w[-rank:])
    cov1_white = W.T.dot(cov1).dot(W)
    eigvals, eigvect = np.linalg.eigh(cov1_white)
    return (
            np.sort(eigvals)[::-1],
            W.dot(eigvect)[:,np.argsort(eigvals)[::-1]])

def bCSTP(data1, data2, num_iter, t_keep, s_keep):
    n_ch, n_dp, n_trials = data1.shape
    t_keep = np.r_[n_dp,
            np.linspace(t_keep, n_dp, num_iter).astype(int)[::-1]]
    s_keep = np.linspace(s_keep, n_ch, num_iter).astype(int)[::-1]
    T_FILT = [np.eye(n_dp)]
    S_FILT = []
    S_EIGVAL = []
    T_EIGVAL = []
    for i in range(num_iter):
        print('bCSTP-iteration num %d' % (i + 1))
        # obtain spatial filter
        temp1 = np.tensordot(T_FILT[-1][:,:t_keep[i]], data1, axes=(0,1))
        temp2 = np.tensordot(T_FILT[-1][:,:t_keep[i]], data2, axes=(0,1))
        cov1 = np.einsum('ijl, ikl -> jk', temp1, temp1)
        cov2 = np.einsum('ijl, ikl -> jk', temp2, temp2)
        w, v = eigh(cov1, cov2)
        S_FILT.append(v)
        S_EIGVAL.append(w)
        # obtain temporal filter
        temp1 = np.tensordot(S_FILT[-1][:,:s_keep[i]], data1, axes=(0,0))
        temp2 = np.tensordot(S_FILT[-1][:,:s_keep[i]], data2, axes=(0,0))
        cov1 = np.einsum('ijl, ikl -> jk', temp1, temp1)
        cov2 = np.einsum('ijl, ikl -> jk', temp2, temp2)
        w, v = eigh(cov1, cov2)
        T_FILT.append(v)
        T_EIGVAL.append(w)
    return S_EIGVAL, T_EIGVAL, S_FILT, T_FILT[1:]

chan_names = ['FZ', 'F3', 'FC5', 'CZ', 'C3', 'CP5', 'T7', 'CP1']

data_for_visualization = [
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K001/02-K01_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K001/03-K01_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K001/04-K01_stim3.dat'],
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K002/02-K02_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K002/03-K02_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K002/04-K02_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K003/02-K03_stim1_2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K003/03-K03_stim2_2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K003/04-K03_stim3_2.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K004/02-K04_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K004/03-K04_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K004/04-K04_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K005/02-K05_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K005/03-K05_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K005/04-K05_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K006/02-K06_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K006/03-K06_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K006/04-K06_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K007/02-K07_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K007/03-K07_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K007/04-K07_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K008/02-K08_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K008/03-K08_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K008/04-K08_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K009/02-K09_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K009/03-K09_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K009/04-K09_stim3.dat'], 
   ['/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K010/02-K10_stim1.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K010/03-K10_stim2.dat', '/home/christoph/Desktop/Data_Thesis_Analyze/AllData/K010/04-K10_stim3.dat'] 
]

#data_for_visualization = [
#   ['/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K001/02-K01_stim1.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K001/03-K01_stim2.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K001/04-K01_stim3.dat'],
#   ['/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K002/02-K02_stim1.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K002/03-K02_stim2.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K002/04-K02_stim3.dat'], 
#   ['/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K003/02-K03_stim1_2.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K003/03-K03_stim2_2.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K003/04-K03_stim3_2.dat'], 
#   ['/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K004/02-K04_stim1.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K004/03-K04_stim2.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K004/04-K04_stim3.dat'], 
#   ['/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K005/02-K05_stim1.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K005/03-K05_stim2.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K005/04-K05_stim3.dat'], 
#   ['/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K006/02-K06_stim1.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K006/03-K06_stim2.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K006/04-K06_stim3.dat'], 
#   ['/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K007/02-K07_stim1.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K007/03-K07_stim2.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K007/04-K07_stim3.dat'], 
#   ['/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K008/02-K08_stim1.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K008/03-K08_stim2.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K008/04-K08_stim3.dat'], 
#   ['/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K009/02-K09_stim1.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K009/03-K09_stim2.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K009/04-K09_stim3.dat'], 
#   ['/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K010/02-K10_stim1.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K010/03-K10_stim2.dat', '/media/christoph/02A8C167A8C15A35/Masterthesis/Original_Data/K010/04-K10_stim3.dat'] 
#]

### ### ### ### ###
### Data loading
### ### ### ### ###  
idx = int(sys.argv[1])
# idx = 1
data_for_train = data_for_visualization[idx]  
data_kx_stim1 = meet.readBinary(data_for_train[0], num_channels=9, data_type='float8')  
data_kx_stim2 = meet.readBinary(data_for_train[1], num_channels=9, data_type='float8')   
data_kx_stim3 = meet.readBinary(data_for_train[2], num_channels=9, data_type='float8')    

hfSEP_win = [50, 450]
noise_win = [-500, -100]
intrplt_win = [-80, 20]

from matplotlib.font_manager import FontProperties        
font = FontProperties()        
font.set_family('serif')        
font.set_name('Times New Roman')        
font.set_style('oblique')       
font.set_size(15)
ticks = np.arange(hfSEP_win[0] - hfSEP_win[0], hfSEP_win[1] - hfSEP_win[0], 20)
ticklabels = ['%.1f ms' % tick for tick in ((ticks + hfSEP_win[0]) / 10)]

### ### ### ### ###
### Basic preprocessing
### ### ### ### ###  
kx_data_combined = np.append(data_kx_stim1[:,10000:], data_kx_stim2[:,10000:], axis=1)  
if idx == 1:
    kx_data_combined = np.append(kx_data_combined, data_kx_stim3[:,10000:1860000], axis=1) # K002, stim3: Channel-Drift, daher nur 10000:1860000!
else:
    kx_data_combined = np.append(kx_data_combined, data_kx_stim3[:,10000:], axis=1) # K002, stim3: Channel-Drift, daher nur 10000:1860000!
print('Applying outlier rejection!')
kx_data_combined = reject_outlier_seconds_using_lsd(kx_data_combined, chan=0, identifier=idx)
triggers_for_kx_combined = identify_triggers(kx_data_combined[8], 300, min(kx_data_combined[8]))

#00_ToDo worked on pcolormesh of trials prior to filtering (CP5-FZ):
plt.figure(figsize=(18,9))
plt.pcolormesh(normalize_min_max(meet.epochEEG(kx_data_combined, triggers_for_kx_combined, hfSEP_win)[5] - meet.epochEEG(kx_data_combined, triggers_for_kx_combined, hfSEP_win)[0]), vmax=1.0, vmin=0.0)
plt.colorbar()
plt.xlabel('Sample', fontproperties=font)
plt.ylabel('Relative time after Stimulus', fontproperties=font)
plt.title('Visualization of epoched [70, 470] raw data of K00%d;\nvisualization normalized to [0, 1]' % (idx + 1), fontproperties=font)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/christoph/Desktop/Meeting_Gunnar_24_09/Meeting_25_09/Meeting_Monday/test_script_outlier_removed/k00%d/000_k00%d_raw_data_colormesh.png' % (idx + 1, idx + 1))
plt.close('all')

#00.1_ToDo worked on pcolormesh of trials prior to filtering (CP5-FZ):
fig = plt.figure(figsize=(18,9))
plt.subplot(211)
plt.pcolormesh(normalize_min_max(meet.epochEEG(kx_data_combined, triggers_for_kx_combined, hfSEP_win)[5] - meet.epochEEG(kx_data_combined, triggers_for_kx_combined, hfSEP_win)[0]), vmax=0.65, vmin=0.5)
plt.colorbar()
plt.xlabel('Sample', fontproperties=font)
plt.ylabel('Relative time after Stimulus', fontproperties=font)
plt.subplot(212)
plt.plot((meet.epochEEG(kx_data_combined, triggers_for_kx_combined, hfSEP_win)[5] - meet.epochEEG(kx_data_combined, triggers_for_kx_combined, hfSEP_win)[0]).mean(-1))
plt.xlabel('Relative time after Stimulus', fontproperties=font)
plt.ylabel('Average signal', fontproperties=font)
plt.xticks(ticks=ticks, labels=ticklabels)
fig.suptitle('Visualization of epoched [%d, %d] raw data of K00%d and the average hfSEP in CP5-FZ;\nnormalized to [0, 1]; zoomed in to [0.25, 0.75]' % (hfSEP_win[0], hfSEP_win[1], idx + 1), fontproperties=font)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/christoph/Desktop/Meeting_Gunnar_24_09/Meeting_25_09/Meeting_Monday/test_script_outlier_removed/k00%d/001_k00%d_raw_data_cp5-fz_colormesh.png' % (idx + 1, idx + 1))
plt.close('all')

intrplt_kx_data_combined = meet.basic.interpolateEEG(kx_data_combined[:8], triggers_for_kx_combined, intrplt_win)
intrplt_kx_data_combined = np.append(intrplt_kx_data_combined, np.expand_dims(kx_data_combined[8], axis=0), axis=0)
intrplt_filt_500_900 = meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=[500, 900], fs=[450, 1000], s_rate=10000)
intrplt_filt_500_900 = np.append(intrplt_filt_500_900, np.expand_dims(intrplt_kx_data_combined[8], axis=0), axis=0)
epchd_intrplt_500_900_hfsep = meet.epochEEG(intrplt_filt_500_900, triggers_for_kx_combined, hfSEP_win)
epchd_intrplt_500_900_noise = meet.epochEEG(intrplt_filt_500_900, triggers_for_kx_combined, noise_win)

#00.1_ToDo worked on pcolormesh of trials prior to filtering (CP5-FZ):
fig = plt.figure(figsize=(18,9))
plt.subplot(211)
plt.pcolormesh(normalize_min_max(
    epchd_intrplt_500_900_hfsep[5] - epchd_intrplt_500_900_hfsep[0]), vmax=0.65, vmin=0.5
)
plt.colorbar()
plt.xlabel('Sample', fontproperties=font)
plt.ylabel('Relative time after Stimulus', fontproperties=font)
plt.subplot(212)
plt.plot(
    (epchd_intrplt_500_900_hfsep[5] - epchd_intrplt_500_900_hfsep[0])
    .mean(-1)
)
plt.xlabel('Relative time after Stimulus', fontproperties=font)
plt.ylabel('Average signal', fontproperties=font)
plt.xticks(ticks=ticks, labels=ticklabels)
fig.suptitle('Visualization of interpolated [%d, %d], epoched [%d, %d], bandpass-filtered fp=[500, 900] data of K00%d and the average hfSEP in CP5-FZ;\nnormalized to [0, 1]; zoomed in to [0.25, 0.75]' % (intrplt_win[0], intrplt_win[1], hfSEP_win[0], hfSEP_win[1], idx + 1), fontproperties=font)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/christoph/Desktop/Meeting_Gunnar_24_09/Meeting_25_09/Meeting_Monday/test_script_outlier_removed/k00%d/002_k00%d_intrplt_filt_500_900_data_cp5-fz_colormesh.png' % (idx + 1, idx + 1))
plt.close('all')

s_eigvals, t_eigvals, W, V = bCSTP(epchd_intrplt_500_900_hfsep[:8], epchd_intrplt_500_900_noise[:8], num_iter=15, t_keep=2, s_keep=2)
data_WVfilt = scipy.ndimage.convolve1d(np.dot(W[-1][:,0], intrplt_filt_500_900[:8]), V[-1][:,0][::-1], axis=-1)
WVtrials = meet.epochEEG(data_WVfilt, triggers_for_kx_combined, hfSEP_win)

#00_ToDo worked on temporal and spatial filter eigenvalues
plt.figure(figsize=(18,9))
plt.plot(s_eigvals[-1])
plt.xlabel('Relative filter', fontproperties=font)
plt.ylabel('Eigenvalue of relative filter', fontproperties=font)
plt.title('Visualization of Eigenvalues of the <<spatial-fitler-to-be-applied>> for data of K00%d' % (idx + 1), fontproperties=font)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/christoph/Desktop/Meeting_Gunnar_24_09/Meeting_25_09/Meeting_Monday/test_script_outlier_removed/k00%d/003_k00%d_spat_bcstp_filt_eigenvals.png' % (idx + 1, idx + 1))
plt.close('all')

plt.figure(figsize=(18,9))
plt.plot(t_eigvals[-1])
plt.xlabel('Relative filter', fontproperties=font)
plt.ylabel('Eigenvalue of relative filter', fontproperties=font)
plt.title('Visualization of Eigenvalues of the <<temporal-fitler-to-be-applied>> for data of K00%d' % (idx + 1), fontproperties=font)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/christoph/Desktop/Meeting_Gunnar_24_09/Meeting_25_09/Meeting_Monday/test_script_outlier_removed/k00%d/004_k00%d_temp_bcstp_filt_eigenvals.png' % (idx + 1, idx + 1))
plt.close('all')

#00_ToDo worked on visualization of pseudo-inverse of temporal filter to be applied
plt.figure(figsize=(18,9))
plt.plot(np.linalg.pinv(V[-1])[0])
plt.xlabel('Relative time after Stimulus', fontproperties=font)
plt.ylabel('From Filter expected value in Sample', fontproperties=font)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.title('Visualization of pseudo-inverse of <<temporal-fitler-to-be-applied>> for data of K00%d' % (idx + 1), fontproperties=font)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/christoph/Desktop/Meeting_Gunnar_24_09/Meeting_25_09/Meeting_Monday/test_script_outlier_removed/k00%d/005_k00%d_pinv_of_temp_filt.png' % (idx + 1, idx + 1))
plt.close('all')

#00_ToDo worked on filtered data after bCSTP-spat
plt.figure(figsize=(18,9))
plt.plot(
    meet.epochEEG(
        np.dot(
            W[-1][:,0], 
            intrplt_filt_500_900[:8]
        ), triggers_for_kx_combined, hfSEP_win
    ).mean(-1)
)
plt.xlabel('Relative time after Stimulus', fontproperties=font)
plt.ylabel('Average signal', fontproperties=font)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.title('Visualization of mean-hfSEP after applying spatial filter of bCSTP to interpolated [%d, %d], epoched [%d, %d], bandpass-filtered fp=[500, 900] data of K00%d' % (intrplt_win[0], intrplt_win[1], hfSEP_win[0], hfSEP_win[1], idx + 1), fontproperties=font)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/christoph/Desktop/Meeting_Gunnar_24_09/Meeting_25_09/Meeting_Monday/test_script_outlier_removed/k00%d/006_k00%d_bcstp_spat_filt.png' % (idx + 1, idx + 1))
plt.close('all')

#00_ToDo worked on filtered data after bCSTP-spat and bCSTP-temp
plt.figure(figsize=(18,9))
plt.plot(
    meet.epochEEG(
        scipy.ndimage.convolve1d(
            np.dot(
                W[-1][:,0], 
                intrplt_filt_500_900[:8]
            )
        , V[-1][:,0][::-1], axis=-1)
    , triggers_for_kx_combined, hfSEP_win)
    .mean(-1)
)
plt.xlabel('Relative time after Stimulus', fontproperties=font)
plt.ylabel('Average signal', fontproperties=font)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.title('Visualization of mean-hfSEP after applying bCSTP to interpolated [%d, %d], epoched [%d, %d], bandpass-filtered fp=[500, 900] data of K00%d' % (intrplt_win[0], intrplt_win[1], hfSEP_win[0], hfSEP_win[1], idx + 1), fontproperties=font)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/christoph/Desktop/Meeting_Gunnar_24_09/Meeting_25_09/Meeting_Monday/test_script_outlier_removed/k00%d/007_k00%d_bcstp_spat_temp_filt.png' % (idx + 1, idx + 1))
plt.close('all')

#00_ToDo worked on pcolormesh after bcstp
plt.figure(figsize=(18,9))
plt.pcolormesh(normalize_min_max(
    meet.epochEEG(
        scipy.ndimage.convolve1d(
            np.dot(
                W[-1][:,0], 
                intrplt_filt_500_900[:8]
            )
        , V[-1][:,0][::-1], axis=-1)
    , triggers_for_kx_combined, hfSEP_win)), vmax=0.75, vmin=0.25)
plt.colorbar()
plt.xlabel('Sample', fontproperties=font)
plt.ylabel('Relative time after Stimulus', fontproperties=font)
plt.title('Visualization of epoched [%d, %d] data of K00%d after applying spatial and temporal filters of bCSTP;\nnormalized to [0, 1]; zoomed in to [0.25, 0.75]' % (hfSEP_win[0], hfSEP_win[1], idx + 1), fontproperties=font)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/christoph/Desktop/Meeting_Gunnar_24_09/Meeting_25_09/Meeting_Monday/test_script_outlier_removed/k00%d/008_k00%d_colormesh_bcstp_spat_temp_filt.png' % (idx + 1, idx + 1))
plt.close('all')

#00_ToDo worked on pcolormesh of squared bcstp
plt.figure(figsize=(18,9))
plt.pcolormesh(normalize_min_max(
    meet.epochEEG(
        scipy.ndimage.convolve1d(
            np.dot(
                W[-1][:,0], 
                intrplt_filt_500_900[:8]
            )
        , V[-1][:,0][::-1], axis=-1)
    , triggers_for_kx_combined, hfSEP_win)**2), vmax=0.05, vmin=0.0
)
plt.colorbar()
plt.xlabel('Sample', fontproperties=font)
plt.ylabel('Relative time after Stimulus', fontproperties=font)
plt.title('Visualization of squared epoched [%d, %d] data of K00%d after applying spatial and temporal filters of bCSTP\nnormalized to [0, 1]; zoomed in to [0.0, 0.05]' % (hfSEP_win[0], hfSEP_win[1], idx + 1), fontproperties=font)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/christoph/Desktop/Meeting_Gunnar_24_09/Meeting_25_09/Meeting_Monday/test_script_outlier_removed/k00%d/009_k00%d_colormesh_squared_bcstp_spat_temp_filt.png' % (idx + 1, idx + 1))
plt.close('all')

#00.2_ToDo worked on filtered data after bCSTP-spat
fig = plt.figure(figsize=(18,9))
plt.subplot(211)
plt.pcolormesh(normalize_min_max(
    meet.epochEEG(
        np.dot(
            W[-1][:,0], 
            intrplt_filt_500_900[:8]
        ), triggers_for_kx_combined, hfSEP_win
    )
    ), vmax=0.55, vmin=0.5
)
plt.colorbar()
plt.xlabel('Sample', fontproperties=font)
plt.ylabel('Relative time after Stimulus', fontproperties=font)
plt.subplot(212)
plt.plot(
    meet.epochEEG(
        np.dot(
            W[-1][:,0], 
            intrplt_filt_500_900[:8]
        ), triggers_for_kx_combined, hfSEP_win
    ).mean(-1)
)
plt.xlabel('Relative time after Stimulus', fontproperties=font)
plt.ylabel('Average signal', fontproperties=font)
plt.xticks(ticks=ticks, labels=ticklabels)
fig.suptitle('Visualization of mean-hfSEP after applying spatial filter of bCSTP to interpolated [%d, %d], epoched [%d, %d], bandpass-filtered fp=[500, 900] data of K00%d\nnormalized to [0, 1]; zoomed in to [0.5, 0.55]' % (intrplt_win[0], intrplt_win[1], hfSEP_win[0], hfSEP_win[1], idx + 1), fontproperties=font)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/christoph/Desktop/Meeting_Gunnar_24_09/Meeting_25_09/Meeting_Monday/test_script_outlier_removed/k00%d/010_k00%d_colormesh_bcstp_spat_filt.png' % (idx + 1, idx + 1))
plt.close('all')

#00.3_ToDo worked on filtered data after temp-spat bCSTP
fig = plt.figure(figsize=(18,9))
plt.subplot(211)
plt.pcolormesh(normalize_min_max(
    meet.epochEEG(
        scipy.ndimage.convolve1d(
            np.dot(
                W[-1][:,0], 
                intrplt_filt_500_900[:8]
            )
        , V[-1][:,0][::-1], axis=-1)
    , triggers_for_kx_combined, hfSEP_win)), vmax=0.55, vmin=0.5
)
plt.colorbar()
plt.xlabel('Sample', fontproperties=font)
plt.ylabel('Relative time after Stimulus', fontproperties=font)
plt.subplot(212)
plt.plot(
    meet.epochEEG(
        scipy.ndimage.convolve1d(
            np.dot(
                W[-1][:,0], 
                intrplt_filt_500_900[:8]
            )
        , V[-1][:,0][::-1], axis=-1)
    , triggers_for_kx_combined, hfSEP_win)
    .mean(-1)
)
plt.xlabel('Relative time after Stimulus', fontproperties=font)
plt.ylabel('Average signal', fontproperties=font)
plt.xticks(ticks=ticks, labels=ticklabels)
fig.suptitle('Visualization of epoched [%d, %d] data of K00%d after applying spatial and temporal filters of bCSTP\nnormalized to [0, 1]; zoomed in to [0.5, 0.55]' % (hfSEP_win[0], hfSEP_win[1], idx + 1), fontproperties=font)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/christoph/Desktop/Meeting_Gunnar_24_09/Meeting_25_09/Meeting_Monday/test_script_outlier_removed/k00%d/011_k00%d_colormesh_bcstp_temp_spat_filt.png' % (idx + 1, idx + 1))
plt.close('all')

#00.3_ToDo worked on squared filtered data after temp-spat bCSTP
fig = plt.figure(figsize=(18,9))
plt.subplot(211)
plt.pcolormesh(normalize_min_max(
    meet.epochEEG(
        scipy.ndimage.convolve1d(
            np.dot(
                W[-1][:,0], 
                intrplt_filt_500_900[:8]
            )
        , V[-1][:,0][::-1], axis=-1)
    , triggers_for_kx_combined, hfSEP_win)**2), vmax=0.0004, vmin=0.0002
)
plt.colorbar()
plt.xlabel('Sample', fontproperties=font)
plt.ylabel('Relative time after Stimulus', fontproperties=font)
plt.subplot(212)
plt.plot(
    meet.epochEEG(
        scipy.ndimage.convolve1d(
            np.dot(
                W[-1][:,0], 
                intrplt_filt_500_900[:8]
            )
        , V[-1][:,0][::-1], axis=-1)
    , triggers_for_kx_combined, hfSEP_win)
    .mean(-1)**2
)
plt.xlabel('Relative time after Stimulus', fontproperties=font)
plt.ylabel('Average signal', fontproperties=font)
plt.xticks(ticks=ticks, labels=ticklabels)
fig.suptitle('Visualization of squared epoched [%d, %d] data of K00%d after applying spatial and temporal filters of bCSTP\nnormalized to [0, 1]; zoomed in to [0.0002, 0.0004]' % (hfSEP_win[0], hfSEP_win[1], idx + 1), fontproperties=font)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/home/christoph/Desktop/Meeting_Gunnar_24_09/Meeting_25_09/Meeting_Monday/test_script_outlier_removed/k00%d/012_k00%d_colormesh_squared_bcstp_temp_spat_filt.png' % (idx + 1, idx + 1))
plt.close('all')

#plt.pcolormesh(normalize_min_max((WVtrials**2), vmax=0.003)
#plt.colorbar()
#plt.figure(figsize=(18,9))
#plt.plot(WVtrials.mean(-1))

# ToDo's bis Montag:
#00_ToDo: Extend w. mean response plotted
#01_ToDo: Put into script for data prep, once checked that the above works for all the subjects!
#02_ToDo: SNNR-Berechnungen und Varianz-Ratio Berechnungen für ein Subjekt erneut durchlaufen lassen und kontrollieren! je Sliding-Window-Avg. berechnen! Daraus Plots erstellen von SNNR vs. Var-Ratios, vs Var-hfSEP vs Var-Noise, vs. wie sieht ein mean-hfSEP und wie sieht eine mean-noise in diesem Fenster aus?
#03_ToDo: SNNR-Berechnungen und Varianz-Ratio Berechnungen für alle Subjekte erneut durchlaufen lassen und je Sliding-Window-Avg. berechnen! Daraus Plots erstellen von SNNR vs. Var-Ratios, vs Var-hfSEP vs Var-Noise, vs. wie sieht ein mean-hfSEP und wie sieht eine mean-noise in diesem Fenster aus?
#04_ToDo: Data-Preprocessing erneut und final durchlaufen lassen, sodass die Daten für jedes Subjekt in allen Formaten vorliegen!
#05_ToDo: Experimente planen, aber genauer! Einen Zeitplan erstellen und probehalber je Architektur ein Experiment starten und bis zur Konvergenz laufen lassen!
#06_ToDo: Den roten Faden in die Masterthesis einarbeiten und wissen, was ich wo schreiben möchte...



### ### ### ### ###
### Outlier-Rejection, filtering and epoching
### ### ### ### ###  
# kx_data_intrplt_outliers_rejected = reject_outlier_seconds_using_lsd(intrplt_kx_data_combined, chan=0, identifier=idx)
# triggers_identified_after_rejection = identify_triggers(kx_data_intrplt_outliers_rejected[8], 300, min(kx_data_intrplt_outliers_rejected[8]))
# intrplt_filt_500_900_kx_out_rej = meet.iir.butterworth(kx_data_intrplt_outliers_rejected[:8], fp=[500, 900], fs=[450, 1000], s_rate=10000)
# intrplt_filt_500_900_kx_out_rej = np.append(intrplt_filt_500_900_kx_out_rej, np.expand_dims(kx_data_intrplt_outliers_rejected[8], axis=0), axis=0)
# epoched_intrplt_filt_500_900_kx_hfsep_out_rej = meet.epochEEG(intrplt_filt_500_900_kx_out_rej, triggers_identified_after_rejection, [100, 300])
# epoched_intrplt_filt_500_900_kx_noise_out_rej = meet.epochEEG(intrplt_filt_500_900_kx_out_rej, triggers_identified_after_rejection, [-300, -100])

#print(epoched_intrplt_filt_500_900_kx_hfsep_out_rej.shape) # ==> (9, 200, 5151)
#epoched_intrplt_filt_500_900_80ms_long_window = meet.epochEEG(intrplt_filt_500_900_kx_out_rej, triggers_identified_after_rejection, [0, 800])

#W_out_500_900, V_out_500_900, s_out_500_900, t_out_500_900 = bCSTP(epoched_intrplt_filt_500_900_kx_hfsep_out_rej[:8,:,:], epoched_intrplt_filt_500_900_kx_noise_out_rej[:8,:,:], num_iter=10)
#trials_dotted_w_V_out = [np.dot(V_out_500_900[-1][:,0], np.tensordot(W_out_500_900[-1][:,0], epoched_intrplt_filt_500_900_80ms_long_window[:8], axes=(0,0))[i:i+200])  for i in range(600)]
#plt.plot(np.array(trials_dotted_w_V_out).T.mean(0), linewidth=4, label='bCSTP-test_0')
