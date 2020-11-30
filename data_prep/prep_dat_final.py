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
offset = 1000
s_rate = 10000
stim_per_sec = 4
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

def trim_data_to_contain_only_valid_seconds(data_to_trim, triggers_of_data_to_trim):
  """
  General Idea:
  In the underlying data the subjects were stimulated by 4Hz.
  The window around a stimulus that we're interested in is 150ms long: 
  [<<-55ms Noise? -10ms>> <<-8ms Intrplt 2ms>> <<5ms hfSEP 45ms>> <<45ms Noise? 95ms>>]:= 150ms
  This leaves us with ~50ms of unused data around each end of our window of interest. (1s / 4)
  If we define the offset to be so that we place the start of data existing in the non-used
  window our seconds-segments will always include the whole window we are interested in, 
  even after outlier-rejection and then re-calculating the trigger-points.
  We only have to pay attention and perform this step everytime first, meaning:
  after filtering but prior to combining the data of different measurements per subject together
  
  Attention:
  Makes use of global variables offset and s_rate!

  Sanity-Checks:
  Check if the length of data returned can be divided by s_rate w.o. rest.
  Or plot some parts of the data, including the trigger-channel in steps of s_rate and see triggers overlaid
  """

  if (triggers_of_data_to_trim[0] - offset) > 0:
    timestamp_last_trigger_to_use_safely = triggers_of_data_to_trim[np.arange(0, len(triggers_of_data_to_trim), 4)[-1]]
    return data_to_trim[:,triggers_of_data_to_trim[0] - offset : timestamp_last_trigger_to_use_safely - offset]  
  else:
    timestamp_last_trigger_to_use_safely = triggers_of_data_to_trim[np.arange(1, len(triggers_of_data_to_trim), 4)[-1]]
    return data_to_trim[:,triggers_of_data_to_trim[1] - offset : timestamp_last_trigger_to_use_safely - offset]

def get_indices_for_noise(triggers_to_get_indices_for):
  """
  General Idea:
  """
  possible_neg_noise_starting_indices = np.arange(-550, -500, 1)
  possible_pos_starting_indices = np.arange(450, 550)
  possible_noise_indices_per_sample = np.append(possible_neg_noise_starting_indices, possible_pos_starting_indices)
  return np.asarray(triggers_to_get_indices_for) + np.random.choice(possible_noise_indices_per_sample, len(triggers_to_get_indices_for))

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
        euclidean_distances = np.sqrt(((lsd_baseline - pxx_per_sec.mean(1))**2).sum(1)) # 0_==sec; 1_==freq
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

### ### ### ### ###
### Data loading
### ### ### ### ###
hfSEP_win = [50, 450]
noise_win = [-500, -100]
intrplt_win = [-80, 30]

idx = int(sys.argv[1])
print('Running final data prepping for k00%d ...' % (idx + 1))
data_for_train = data_for_visualization[idx]  

data_kx_stim1 = meet.readBinary(data_for_train[0], num_channels=9, data_type='float8')
data_kx_stim2 = meet.readBinary(data_for_train[1], num_channels=9, data_type='float8') 

# K002, stim3: Channel-Drift, daher nur 10000:1860000!
if idx == 1:
  data_kx_stim3 = meet.readBinary(data_for_train[2], num_channels=9, data_type='float8')[:,10000:1860000]
else:  
  data_kx_stim3 = meet.readBinary(data_for_train[2], num_channels=9, data_type='float8')

triggers_for_data_kx_stim1 = identify_triggers(data_kx_stim1[8], 300, min(data_kx_stim1[8]))
triggers_for_data_kx_stim2 = identify_triggers(data_kx_stim2[8], 300, min(data_kx_stim2[8]))
triggers_for_data_kx_stim3 = identify_triggers(data_kx_stim3[8], 300, min(data_kx_stim3[8]))

trimmed_data_kx_stim1 = trim_data_to_contain_only_valid_seconds(data_kx_stim1, triggers_for_data_kx_stim1)
trimmed_data_kx_stim2 = trim_data_to_contain_only_valid_seconds(data_kx_stim2, triggers_for_data_kx_stim2)
trimmed_data_kx_stim3 = trim_data_to_contain_only_valid_seconds(data_kx_stim3, triggers_for_data_kx_stim3)

### ### ### ### ###
### Basic preprocessing for all the datasets needed
### ### ### ### ###  
kx_data_combined = np.append(trimmed_data_kx_stim1, trimmed_data_kx_stim2, axis=1)  
kx_data_combined = np.append(kx_data_combined, trimmed_data_kx_stim3, axis=1)
save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/kx_data_combined' % (idx + 1), kx_data_combined)

triggers_for_kx_combined = identify_triggers(kx_data_combined[8], 300, min(kx_data_combined[8]))
save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined' % (idx + 1), triggers_for_kx_combined)

intrplt_kx_data_combined = meet.basic.interpolateEEG(kx_data_combined[:8], triggers_for_kx_combined, intrplt_win)
intrplt_kx_data_combined = np.append(intrplt_kx_data_combined, np.expand_dims(kx_data_combined[8], axis=0), axis=0)
save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_kx_data_combined' % (idx + 1), intrplt_kx_data_combined)

### ### ### ### ###
### Data loading in two separate functions to save on RAM
### ### ### ### ###
def prep_without_outlier_rejection():

  ### ### ### ### ###
  ### Basic preprocessing without outlier-rejection
  ### ### ### ### ###
  intrplt_filt_under_100_kx = meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=100, fs=110, s_rate=10000)
  intrplt_filt_under_100_kx = np.append(intrplt_filt_under_100_kx, np.expand_dims(intrplt_kx_data_combined[8], axis=0), axis=0)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_under_100_kx' % (idx + 1), intrplt_filt_under_100_kx)

  intrplt_filt_over_100_kx = meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=100, fs=90, s_rate=10000)
  intrplt_filt_over_100_kx = np.append(intrplt_filt_over_100_kx, np.expand_dims(intrplt_kx_data_combined[8], axis=0), axis=0)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_100_kx' % (idx + 1), intrplt_filt_over_100_kx)

  intrplt_filt_over_400_kx = meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=400, fs=360, s_rate=10000)
  intrplt_filt_over_400_kx = np.append(intrplt_filt_over_400_kx, np.expand_dims(intrplt_kx_data_combined[8], axis=0), axis=0)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_400_kx' % (idx + 1), intrplt_filt_over_400_kx)

  intrplt_filt_500_900_kx = meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=[500, 900], fs=[450, 1000], s_rate=10000)
  intrplt_filt_500_900_kx = np.append(intrplt_filt_500_900_kx, np.expand_dims(intrplt_kx_data_combined[8], axis=0), axis=0)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx' % (idx + 1), intrplt_filt_500_900_kx)

  epoched_kx_data_combined_hfsep = meet.epochEEG(kx_data_combined, triggers_for_kx_combined, hfSEP_win)
  epoched_kx_data_combined_noise = meet.epochEEG(kx_data_combined, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  print(epoched_kx_data_combined_hfsep.shape)
  print(epoched_kx_data_combined_noise.shape)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep' % (idx + 1), epoched_kx_data_combined_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise' % (idx + 1), epoched_kx_data_combined_noise)

  epoched_intrplt_kx_data_combined_hfsep = meet.epochEEG(intrplt_kx_data_combined, triggers_for_kx_combined, hfSEP_win)
  epoched_intrplt_kx_data_combined_noise = meet.epochEEG(intrplt_kx_data_combined, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep' % (idx + 1), epoched_intrplt_kx_data_combined_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise' % (idx + 1), epoched_intrplt_kx_data_combined_noise)

  epoched_intrplt_filt_under_100_kx_hfsep = meet.epochEEG(intrplt_filt_under_100_kx, triggers_for_kx_combined, hfSEP_win)
  epoched_intrplt_filt_under_100_kx_noise = meet.epochEEG(intrplt_filt_under_100_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep' % (idx + 1), epoched_intrplt_filt_under_100_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise' % (idx + 1), epoched_intrplt_filt_under_100_kx_noise)

  epoched_intrplt_filt_over_100_kx_hfsep = meet.epochEEG(intrplt_filt_over_100_kx, triggers_for_kx_combined, hfSEP_win)
  epoched_intrplt_filt_over_100_kx_noise = meet.epochEEG(intrplt_filt_over_100_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep' % (idx + 1), epoched_intrplt_filt_over_100_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise' % (idx + 1), epoched_intrplt_filt_over_100_kx_noise)

  epoched_intrplt_filt_over_400_kx_hfsep = meet.epochEEG(intrplt_filt_over_400_kx, triggers_for_kx_combined, hfSEP_win)
  epoched_intrplt_filt_over_400_kx_noise = meet.epochEEG(intrplt_filt_over_400_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep' % (idx + 1), epoched_intrplt_filt_over_400_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise' % (idx + 1), epoched_intrplt_filt_over_400_kx_noise)

  epoched_intrplt_filt_500_900_kx_hfsep = meet.epochEEG(intrplt_filt_500_900_kx, triggers_for_kx_combined, hfSEP_win)
  epoched_intrplt_filt_500_900_kx_noise = meet.epochEEG(intrplt_filt_500_900_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep' % (idx + 1), epoched_intrplt_filt_500_900_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise' % (idx + 1), epoched_intrplt_filt_500_900_kx_noise)

  print('For k00%d basic preprocessing without outlier-rejection made' % (idx + 1))

  ### ### ### ### ###
  ### Z-Normalization of Basic Preprocessing without outlier-rejection
  ### ### ### ### ###

  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/intrplt_kx_data_combined' % (idx + 1), normalize_z(intrplt_kx_data_combined))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/intrplt_filt_under_100_kx' % (idx + 1), normalize_z(intrplt_filt_under_100_kx))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/intrplt_filt_over_100_kx' % (idx + 1), normalize_z(intrplt_filt_over_100_kx))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/intrplt_filt_over_400_kx' % (idx + 1), normalize_z(intrplt_filt_over_400_kx))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/intrplt_filt_500_900_kx' % (idx + 1), normalize_z(intrplt_filt_500_900_kx))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_kx_data_combined_hfsep' % (idx + 1), normalize_z(epoched_kx_data_combined_hfsep))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_kx_data_combined_noise' % (idx + 1), normalize_z(epoched_kx_data_combined_noise))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_kx_data_combined_hfsep' % (idx + 1), normalize_z(epoched_intrplt_kx_data_combined_hfsep))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_kx_data_combined_noise' % (idx + 1), normalize_z(epoched_intrplt_kx_data_combined_noise))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_under_100_kx_hfsep' % (idx + 1), normalize_z(epoched_intrplt_filt_under_100_kx_hfsep))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_under_100_kx_noise' % (idx + 1), normalize_z(epoched_intrplt_filt_under_100_kx_noise))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_over_100_kx_hfsep' % (idx + 1), normalize_z(epoched_intrplt_filt_over_100_kx_hfsep))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_over_100_kx_noise' % (idx + 1), normalize_z(epoched_intrplt_filt_over_100_kx_noise))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_over_400_kx_hfsep' % (idx + 1), normalize_z(epoched_intrplt_filt_over_400_kx_hfsep))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_over_400_kx_noise' % (idx + 1), normalize_z(epoched_intrplt_filt_over_400_kx_noise))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_500_900_kx_hfsep' % (idx + 1), normalize_z(epoched_intrplt_filt_500_900_kx_hfsep))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized/k00%d/epoched_intrplt_filt_500_900_kx_noise' % (idx + 1), normalize_z(epoched_intrplt_filt_500_900_kx_noise))

  print('For k00%d z-normalization of basic preprocessing without outlier-rejection made' % (idx + 1))

  ### ### ### ### ###
  ### Advanced Preprocessing on data without outlier-rejection
  ### ### ### ### ###

  # CSP is the signal decomposition using two different signal modalities, due to different points in time, but same preprocessing
  print('Shape of epoched_kx_data_combined_hfsep:')
  print(epoched_kx_data_combined_hfsep.shape)
  csp_epoched_kx_data_combined_hfsep_filters, csp_epoched_kx_data_combined_hfsep_eigenvals = meet.spatfilt.CSP(epoched_kx_data_combined_hfsep[:8].reshape(8, -1), epoched_kx_data_combined_noise[:8].reshape(8, -1))
  csp_epoched_intrplt_kx_data_combined_hfsep_filters, csp_epoched_intrplt_kx_data_combined_hfsep_eigenvals = meet.spatfilt.CSP(epoched_intrplt_kx_data_combined_hfsep[:8].reshape(8, -1), epoched_intrplt_kx_data_combined_noise[:8].reshape(8, -1))
  csp_epoched_intrplt_filt_under_100_kx_filters, csp_epoched_intrplt_filt_under_100_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_under_100_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_under_100_kx_noise[:8].reshape(8, -1))
  csp_epoched_intrplt_filt_over_100_kx_filters, csp_epoched_intrplt_filt_over_100_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_over_100_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_over_100_kx_noise[:8].reshape(8, -1))
  csp_epoched_intrplt_filt_over_400_kx_filters, csp_epoched_intrplt_filt_over_400_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_over_400_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_over_400_kx_noise[:8].reshape(8, -1))
  csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))

  csp_filt_epoched_kx_data_combined = np.tensordot(csp_epoched_kx_data_combined_hfsep_filters[:,0].T, kx_data_combined[:8], axes=(0, 0))
  csp_filt_epoched_intrplt_kx_data_combined = np.tensordot(csp_epoched_intrplt_kx_data_combined_hfsep_filters[:,0].T, intrplt_kx_data_combined[:8], axes=(0, 0))
  csp_filt_epoched_intrplt_filt_under_100_kx = np.tensordot(csp_epoched_intrplt_filt_under_100_kx_filters[:,0].T, intrplt_filt_under_100_kx[:8], axes=(0, 0))
  csp_filt_epoched_intrplt_filt_over_100_kx = np.tensordot(csp_epoched_intrplt_filt_over_100_kx_filters[:,0].T, intrplt_filt_over_100_kx[:8], axes=(0, 0))
  csp_filt_epoched_intrplt_filt_over_400_kx = np.tensordot(csp_epoched_intrplt_filt_over_400_kx_filters[:,0].T, intrplt_filt_over_400_kx[:8], axes=(0, 0))
  csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))

  csp_filt_epoched_kx_data_combined_hfsep = meet.epochEEG(csp_filt_epoched_kx_data_combined, triggers_for_kx_combined, hfSEP_win)
  csp_filt_epoched_intrplt_kx_data_combined_hfsep = meet.epochEEG(csp_filt_epoched_intrplt_kx_data_combined, triggers_for_kx_combined, hfSEP_win)
  csp_filt_epoched_intrplt_filt_under_100_kx_hfsep = meet.epochEEG(csp_filt_epoched_intrplt_filt_under_100_kx, triggers_for_kx_combined, hfSEP_win)
  csp_filt_epoched_intrplt_filt_over_100_kx_hfsep = meet.epochEEG(csp_filt_epoched_intrplt_filt_over_100_kx, triggers_for_kx_combined, hfSEP_win)
  csp_filt_epoched_intrplt_filt_over_400_kx_hfsep = meet.epochEEG(csp_filt_epoched_intrplt_filt_over_400_kx, triggers_for_kx_combined, hfSEP_win)
  csp_filt_epoched_intrplt_filt_500_900_kx_hfsep = meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_for_kx_combined, hfSEP_win)

  csp_filt_epoched_kx_data_combined_noise = meet.epochEEG(csp_filt_epoched_kx_data_combined, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  csp_filt_epoched_intrplt_kx_data_combined_noise = meet.epochEEG(csp_filt_epoched_intrplt_kx_data_combined, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  csp_filt_epoched_intrplt_filt_under_100_kx_noise = meet.epochEEG(csp_filt_epoched_intrplt_filt_under_100_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  csp_filt_epoched_intrplt_filt_over_100_kx_noise = meet.epochEEG(csp_filt_epoched_intrplt_filt_over_100_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  csp_filt_epoched_intrplt_filt_over_400_kx_noise = meet.epochEEG(csp_filt_epoched_intrplt_filt_over_400_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  csp_filt_epoched_intrplt_filt_500_900_kx_noise = meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)

  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_kx_data_combined_hfsep' % (idx + 1), csp_filt_epoched_kx_data_combined_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_intrplt_kx_data_combined_hfsep' % (idx + 1), csp_filt_epoched_intrplt_kx_data_combined_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_intrplt_filt_under_100_kx_hfsep' % (idx + 1), csp_filt_epoched_intrplt_filt_under_100_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_intrplt_filt_over_100_kx_hfsep' % (idx + 1), csp_filt_epoched_intrplt_filt_over_100_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_intrplt_filt_over_400_kx_hfsep' % (idx + 1), csp_filt_epoched_intrplt_filt_over_400_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep' % (idx + 1), csp_filt_epoched_intrplt_filt_500_900_kx_hfsep)

  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_kx_data_combined_noise' % (idx + 1), csp_filt_epoched_kx_data_combined_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_intrplt_kx_data_combined_noise' % (idx + 1), csp_filt_epoched_intrplt_kx_data_combined_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_intrplt_filt_under_100_kx_noise' % (idx + 1), csp_filt_epoched_intrplt_filt_under_100_kx_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_intrplt_filt_over_100_kx_noise' % (idx + 1), csp_filt_epoched_intrplt_filt_over_100_kx_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_intrplt_filt_over_400_kx_noise' % (idx + 1), csp_filt_epoched_intrplt_filt_over_400_kx_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise' % (idx + 1), csp_filt_epoched_intrplt_filt_500_900_kx_noise)

  print('For k00%d CSP without outlier-rejection made' % (idx + 1))
  # SSD is the signal decomposition using two different frequency-filtered datasets as different conditions
  
  print('Skipping SSD for now')
######  ssd_intrplt_filt_under_100_kx_filters, ssd_intrplt_filt_under_100_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_under_100_kx[:8], intrplt_kx_data_combined[:8])
######  ssd_intrplt_filt_over_100_kx_filters, ssd_intrplt_filt_over_100_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_over_100_kx[:8], intrplt_kx_data_combined[:8])
######  ssd_intrplt_filt_over_400_kx_filters, ssd_intrplt_filt_over_400_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_over_400_kx[:8], intrplt_kx_data_combined[:8])
######  ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx[:8], intrplt_kx_data_combined[:8])
######
######  ssd_filt_intrplt_filt_under_100_kx = ssd_intrplt_filt_under_100_kx_filters[:,0].T.dot(intrplt_filt_under_100_kx[:8])
######  ssd_filt_intrplt_filt_over_100_kx = ssd_intrplt_filt_over_100_kx_filters.T.dot(intrplt_filt_over_100_kx[:8])
######  ssd_filt_intrplt_filt_over_400_kx_kx = ssd_intrplt_filt_over_400_kx_filters.T.dot(intrplt_filt_over_400_kx[:8])
######  ssd_filt_intrplt_filt_500_900_kx = ssd_intrplt_filt_500_900_kx_filters.T.dot(intrplt_filt_500_900_kx[:8])
######
######  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_under_100_kx' % (idx + 1), ssd_filt_intrplt_filt_under_100_kx)
######  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_100_kx' % (idx + 1), ssd_filt_intrplt_filt_over_100_kx)
######  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_400_kx_kx' % (idx + 1), ssd_filt_intrplt_filt_over_400_kx_kx)
######  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_500_900_kx' % (idx + 1), ssd_filt_intrplt_filt_500_900_kx)

  print('For k00%d SSD without outlier-rejection made' % (idx + 1))
  # CCAr is the technique that tries to derive filters that 'modify' the single-trial to be more similar to the single-trial averages
  a_epoched_kx_data_combined_hfsep, b_epoched_kx_data_combined_hfsep, s_epoched_kx_data_combined_hfsep = meet.spatfilt.CCAvReg(epoched_kx_data_combined_hfsep[:8,:,:])
  a_epoched_intrplt_kx_data_combined_hfsep, b_epoched_intrplt_kx_data_combined_hfsep, s_epoched_intrplt_kx_data_combined_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_kx_data_combined_hfsep[:8,:,:])
  a_epoched_intrplt_filt_under_100_kx_hfsep, b_epoched_intrplt_filt_under_100_kx_hfsep, s_epoched_intrplt_filt_under_100_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_under_100_kx_hfsep[:8,:,:])
  a_epoched_intrplt_filt_over_100_kx_hfsep, b_epoched_intrplt_filt_over_100_kx_hfsep, s_epoched_intrplt_filt_over_100_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_over_100_kx_hfsep[:8,:,:])
  a_epoched_intrplt_filt_over_400_kx_hfsep, b_epoched_intrplt_filt_over_400_kx_hfsep, s_epoched_intrplt_filt_over_400_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_over_400_kx_hfsep[:8,:,:])
  a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])

  ccar_filt_epoched_kx_data_combined_hfsep = np.tensordot(a_epoched_kx_data_combined_hfsep[:,0], epoched_kx_data_combined_hfsep[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_kx_data_combined_noise = np.tensordot(a_epoched_kx_data_combined_hfsep[:,0], epoched_kx_data_combined_noise[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_kx_data_combined_hfsep = np.tensordot(a_epoched_intrplt_kx_data_combined_hfsep[:,0], epoched_intrplt_kx_data_combined_hfsep[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_kx_data_combined_noise = np.tensordot(a_epoched_intrplt_kx_data_combined_hfsep[:,0], epoched_intrplt_kx_data_combined_noise[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_under_100_kx_hfsep = np.tensordot(a_epoched_intrplt_filt_under_100_kx_hfsep[:,0], epoched_intrplt_filt_under_100_kx_hfsep[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_under_100_kx_noise = np.tensordot(a_epoched_intrplt_filt_under_100_kx_hfsep[:,0], epoched_intrplt_filt_under_100_kx_noise[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_over_100_kx_hfsep = np.tensordot(a_epoched_intrplt_filt_over_100_kx_hfsep[:,0], epoched_intrplt_filt_over_100_kx_hfsep[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_over_100_kx_noise = np.tensordot(a_epoched_intrplt_filt_over_100_kx_hfsep[:,0], epoched_intrplt_filt_over_100_kx_noise[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_over_400_kx_hfsep = np.tensordot(a_epoched_intrplt_filt_over_400_kx_hfsep[:,0], epoched_intrplt_filt_over_400_kx_hfsep[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_over_400_kx_noise = np.tensordot(a_epoched_intrplt_filt_over_400_kx_hfsep[:,0], epoched_intrplt_filt_over_400_kx_noise[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_500_900_kx_hfsep = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_500_900_kx_noise = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], epoched_intrplt_filt_500_900_kx_noise[:8,:,:], axes=(0, 0))

  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_kx_data_combined_hfsep' % (idx + 1), ccar_filt_epoched_kx_data_combined_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_kx_data_combined_noise' % (idx + 1), ccar_filt_epoched_kx_data_combined_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_kx_data_combined_hfsep' % (idx + 1), ccar_filt_epoched_intrplt_kx_data_combined_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_kx_data_combined_noise' % (idx + 1), ccar_filt_epoched_intrplt_kx_data_combined_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_under_100_kx_hfsep' % (idx + 1), ccar_filt_epoched_intrplt_filt_under_100_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_under_100_kx_noise' % (idx + 1), ccar_filt_epoched_intrplt_filt_under_100_kx_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_over_100_kx_hfsep' % (idx + 1), ccar_filt_epoched_intrplt_filt_over_100_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_over_100_kx_noise' % (idx + 1), ccar_filt_epoched_intrplt_filt_over_100_kx_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_over_400_kx_hfsep' % (idx + 1), ccar_filt_epoched_intrplt_filt_over_400_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_over_400_kx_noise' % (idx + 1), ccar_filt_epoched_intrplt_filt_over_400_kx_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_500_900_kx_hfsep' % (idx + 1), ccar_filt_epoched_intrplt_filt_500_900_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_epoched_intrplt_filt_500_900_kx_noise' % (idx + 1), ccar_filt_epoched_intrplt_filt_500_900_kx_noise)

  print('For k00%d CCAr without outlier-rejection made' % (idx + 1))
  # bCSPT is the technique that tries to derive filters in the spatial and in the temporal domain, leading to the use of convolution
  s_outepoched_intrplt_kx_data_combined_hfsep_eigenvals, t_outepoched_intrplt_kx_data_combined_hfsep_eigenvals, W_out_epoched_intrplt_kx_data_combined_hfsep, V_out_epoched_intrplt_kx_data_combined_hfsep = bCSTP(epoched_intrplt_kx_data_combined_hfsep[:8,:,:], epoched_intrplt_kx_data_combined_noise[:8,:,:], num_iter=15, t_keep=2, s_keep=2)
  s_outepoched_intrplt_filt_under_100_kx_hfsep_eigenvals, t_outepoched_intrplt_filt_under_100_kx_hfsep_eigenvals, W_out_epoched_intrplt_filt_under_100_kx_hfsep, V_out_epoched_intrplt_filt_under_100_kx_hfsep = bCSTP(epoched_intrplt_filt_under_100_kx_hfsep[:8,:,:], epoched_intrplt_filt_under_100_kx_noise[:8,:,:], num_iter=15, t_keep=2, s_keep=2)
  s_outepoched_intrplt_filt_over_100_kx_hfsep_eigenvals, t_outepoched_intrplt_filt_over_100_kx_hfsep_eigenvals, W_out_epoched_intrplt_filt_over_100_kx_hfsep, V_out_epoched_intrplt_filt_over_100_kx_hfsep = bCSTP(epoched_intrplt_filt_over_100_kx_hfsep[:8,:,:], epoched_intrplt_filt_over_100_kx_noise[:8,:,:], num_iter=15, t_keep=2, s_keep=2)
  s_outepoched_intrplt_filt_over_400_kx_hfsep_eigenvals, t_outepoched_intrplt_filt_over_400_kx_hfsep_eigenvals, W_out_epoched_intrplt_filt_over_400_kx_hfsep, V_out_epoched_intrplt_filt_over_400_kx_hfsep = bCSTP(epoched_intrplt_filt_over_400_kx_hfsep[:8,:,:], epoched_intrplt_filt_over_400_kx_noise[:8,:,:], num_iter=15, t_keep=2, s_keep=2)
  s_outepoched_intrplt_filt_500_900_kx_hfsep_eigenvals, t_outepoched_intrplt_filt_500_900_kx_hfsep_eigenvals, W_out_epoched_intrplt_filt_500_900_kx_hfsep, V_out_epoched_intrplt_filt_500_900_kx_hfsep = bCSTP(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:], epoched_intrplt_filt_500_900_kx_noise[:8,:,:], num_iter=15, t_keep=2, s_keep=2)

  bcstp_spat_temp_filt_epoched_intrplt_kx_data_combined = scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_kx_data_combined_hfsep[-1][:,0], intrplt_kx_data_combined[:8]), V_out_epoched_intrplt_kx_data_combined_hfsep[-1][:,0][::-1], axis=-1)
  bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx = scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_filt_under_100_kx_hfsep[-1][:,0], intrplt_filt_under_100_kx[:8]), V_out_epoched_intrplt_filt_under_100_kx_hfsep[-1][:,0][::-1], axis=-1)
  bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx = scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_filt_over_100_kx_hfsep[-1][:,0], intrplt_filt_over_100_kx[:8]), V_out_epoched_intrplt_filt_over_100_kx_hfsep[-1][:,0][::-1], axis=-1)
  bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx = scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_filt_over_400_kx_hfsep[-1][:,0], intrplt_filt_over_400_kx[:8]), V_out_epoched_intrplt_filt_over_400_kx_hfsep[-1][:,0][::-1], axis=-1)
  bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx = scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_filt_500_900_kx_hfsep[-1][:,0], intrplt_filt_500_900_kx[:8]), V_out_epoched_intrplt_filt_500_900_kx_hfsep[-1][:,0][::-1], axis=-1)

  bcstp_spat_temp_filt_epoched_intrplt_kx_data_combined_hfsep = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_kx_data_combined, triggers_for_kx_combined, hfSEP_win)
  bcstp_spat_temp_filt_epoched_intrplt_kx_data_combined_noise = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_kx_data_combined, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_hfsep = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx, triggers_for_kx_combined, hfSEP_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_noise = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_hfsep = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx, triggers_for_kx_combined, hfSEP_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_noise = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_hfsep = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx, triggers_for_kx_combined, hfSEP_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_noise = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_hfsep = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx, triggers_for_kx_combined, hfSEP_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_noise = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx, get_indices_for_noise(triggers_for_kx_combined), noise_win)

  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_intrplt_kx_data_combined_hfsep' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_kx_data_combined_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_intrplt_kx_data_combined_noise' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_kx_data_combined_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_hfsep' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_noise' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_hfsep' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_noise' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_hfsep' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_noise' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_noise)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_hfsep' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_hfsep)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_noise' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_noise)

  print('For k00%d bCSTP without outlier-rejection made' % (idx + 1))


### ### ### ### ###
### Data loading in two separate functions to save on RAM
### ### ### ### ###
def prep_with_outlier_rejection():

  ### ### ### ### ###
  ### Basic preprocessing with outlier-rejection
  ### ### ### ### ###

  kx_data_intrplt_outliers_rejected = reject_outlier_seconds_using_lsd(intrplt_kx_data_combined, chan=0, identifier=idx)
  triggers_identified_after_rejection = np.arange(offset, len(kx_data_intrplt_outliers_rejected[0]), s_rate / stim_per_sec, dtype=np.int32)
  # kx_data_intrplt_outliers_rejected = np.append(kx_data_intrplt_outliers_rejected, np.expand_dims(triggers_identified_after_rejection, axis=0), axis=0)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/kx_data_intrplt_outliers_rejected' % (idx + 1), kx_data_intrplt_outliers_rejected)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/triggers_identified_after_rejection' % (idx + 1), triggers_identified_after_rejection)

  intrplt_filt_under_100_kx_out_rej = reject_outlier_seconds_using_lsd(meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=100, fs=110, s_rate=10000), chan=0, identifier=idx)
  triggers_identified_after_rejection_under_100_kx_out_rej = np.arange(offset, len(intrplt_filt_under_100_kx_out_rej[0]), s_rate / stim_per_sec, dtype=np.int32)
  # intrplt_filt_under_100_kx_out_rej = np.append(intrplt_filt_under_100_kx, np.expand_dims(triggers_identified_after_rejection_under_100_kx_out_rej, axis=0), axis=0)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/intrplt_filt_under_100_kx_out_rej' % (idx + 1), intrplt_filt_under_100_kx_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/triggers_identified_after_rejection_under_100_kx_out_rej' % (idx + 1), triggers_identified_after_rejection_under_100_kx_out_rej)

  intrplt_filt_over_100_kx_out_rej = reject_outlier_seconds_using_lsd(meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=100, fs=90, s_rate=10000), chan=0, identifier=idx)
  triggers_identified_after_rejection_over_100_kx_out_rej = np.arange(offset, len(intrplt_filt_over_100_kx_out_rej[0]), s_rate / stim_per_sec, dtype=np.int32)
  # intrplt_filt_over_100_kx_out_rej = np.append(intrplt_filt_over_100_kx, np.expand_dims(triggers_identified_after_rejection_over_100_kx_out_rej, axis=0), axis=0)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/intrplt_filt_over_100_kx_out_rej' % (idx + 1), intrplt_filt_over_100_kx_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/triggers_identified_after_rejection_over_100_kx_out_rej' % (idx + 1), triggers_identified_after_rejection_over_100_kx_out_rej)

  intrplt_filt_over_400_kx_out_rej = reject_outlier_seconds_using_lsd(meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=400, fs=360, s_rate=10000), chan=0, identifier=idx)
  triggers_identified_after_rejection_over_400_kx_out_rej = np.arange(offset, len(intrplt_filt_over_400_kx_out_rej[0]), s_rate / stim_per_sec, dtype=np.int32)
  # intrplt_filt_over_400_kx_out_rej = np.append(intrplt_filt_over_400_kx, np.expand_dims(triggers_identified_after_rejection_over_400_kx_out_rej, axis=0), axis=0)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/intrplt_filt_over_400_kx_out_rej' % (idx + 1), intrplt_filt_over_400_kx_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/triggers_identified_after_rejection_over_400_kx_out_rej' % (idx + 1), triggers_identified_after_rejection_over_400_kx_out_rej)

  intrplt_filt_500_900_kx_out_rej = reject_outlier_seconds_using_lsd(meet.iir.butterworth(intrplt_kx_data_combined[:8], fp=[500, 900], fs=[450, 1000], s_rate=10000), chan=0, identifier=idx)
  triggers_identified_after_rejection_500_900_kx_out_rej = np.arange(offset, len(intrplt_filt_500_900_kx_out_rej[0]), s_rate / stim_per_sec, dtype=np.int32)
  # intrplt_filt_500_900_kx_out_rej = np.append(intrplt_filt_500_900_kx, np.expand_dims(triggers_identified_after_rejection_500_900_kx_out_rej, axis=0), axis=0)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/intrplt_filt_500_900_kx_out_rej' % (idx + 1), intrplt_filt_500_900_kx_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/triggers_identified_after_rejection_500_900_kx_out_rej' % (idx + 1), triggers_identified_after_rejection_500_900_kx_out_rej)

  epoched_intrplt_kx_data_combined_hfsep_out_rej = meet.epochEEG(kx_data_intrplt_outliers_rejected, triggers_identified_after_rejection, hfSEP_win)
  epoched_intrplt_kx_data_combined_noise_out_rej = meet.epochEEG(kx_data_intrplt_outliers_rejected, get_indices_for_noise(triggers_identified_after_rejection), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_hfsep_out_rej' % (idx + 1), epoched_intrplt_kx_data_combined_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_noise_out_rej' % (idx + 1), epoched_intrplt_kx_data_combined_noise_out_rej)

  epoched_intrplt_filt_under_100_kx_hfsep_out_rej = meet.epochEEG(intrplt_filt_under_100_kx_out_rej, triggers_identified_after_rejection_under_100_kx_out_rej, hfSEP_win)
  epoched_intrplt_filt_under_100_kx_noise_out_rej = meet.epochEEG(intrplt_filt_under_100_kx_out_rej, get_indices_for_noise(triggers_identified_after_rejection_under_100_kx_out_rej), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_hfsep_out_rej' % (idx + 1), epoched_intrplt_filt_under_100_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_noise_out_rej' % (idx + 1), epoched_intrplt_filt_under_100_kx_noise_out_rej)

  epoched_intrplt_filt_over_100_kx_hfsep_out_rej = meet.epochEEG(intrplt_filt_over_100_kx_out_rej, triggers_identified_after_rejection_over_100_kx_out_rej, hfSEP_win)
  epoched_intrplt_filt_over_100_kx_noise_out_rej = meet.epochEEG(intrplt_filt_over_100_kx_out_rej, get_indices_for_noise(triggers_identified_after_rejection_over_100_kx_out_rej), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_hfsep_out_rej' % (idx + 1), epoched_intrplt_filt_over_100_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_noise_out_rej' % (idx + 1), epoched_intrplt_filt_over_100_kx_noise_out_rej)

  epoched_intrplt_filt_over_400_kx_hfsep_out_rej = meet.epochEEG(intrplt_filt_over_400_kx_out_rej, triggers_identified_after_rejection_over_400_kx_out_rej, hfSEP_win)
  epoched_intrplt_filt_over_400_kx_noise_out_rej = meet.epochEEG(intrplt_filt_over_400_kx_out_rej, get_indices_for_noise(triggers_identified_after_rejection_over_400_kx_out_rej), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_hfsep_out_rej' % (idx + 1), epoched_intrplt_filt_over_400_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_noise_out_rej' % (idx + 1), epoched_intrplt_filt_over_400_kx_noise_out_rej)

  epoched_intrplt_filt_500_900_kx_hfsep_out_rej = meet.epochEEG(intrplt_filt_500_900_kx_out_rej, triggers_identified_after_rejection_500_900_kx_out_rej, hfSEP_win)
  epoched_intrplt_filt_500_900_kx_noise_out_rej = meet.epochEEG(intrplt_filt_500_900_kx_out_rej, get_indices_for_noise(triggers_identified_after_rejection_500_900_kx_out_rej), noise_win)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_hfsep_out_rej' % (idx + 1), epoched_intrplt_filt_500_900_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_noise_out_rej' % (idx + 1), epoched_intrplt_filt_500_900_kx_noise_out_rej)

  print('For k00%d basic preprocessing with outlier-rejection made' % (idx + 1))

  ### ### ### ### ###
  ### Z-Normalization of Basic Preprocessing with outlier-rejection
  ### ### ### ### ###

  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/kx_data_intrplt_outliers_rejected' % (idx + 1), normalize_z(kx_data_intrplt_outliers_rejected))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/intrplt_filt_under_100_kx_out_rej' % (idx + 1), normalize_z(intrplt_filt_under_100_kx_out_rej))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/intrplt_filt_over_100_kx_out_rej' % (idx + 1), normalize_z(intrplt_filt_over_100_kx_out_rej))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/intrplt_filt_over_400_kx_out_rej' % (idx + 1), normalize_z(intrplt_filt_over_400_kx_out_rej))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/intrplt_filt_500_900_kx_out_rej' % (idx + 1), normalize_z(intrplt_filt_500_900_kx_out_rej))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_hfsep_out_rej' % (idx + 1), normalize_z(epoched_intrplt_kx_data_combined_hfsep_out_rej))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_noise_out_rej' % (idx + 1), normalize_z(epoched_intrplt_kx_data_combined_noise_out_rej))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_hfsep_out_rej' % (idx + 1), normalize_z(epoched_intrplt_filt_under_100_kx_hfsep_out_rej))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_noise_out_rej' % (idx + 1), normalize_z(epoched_intrplt_filt_under_100_kx_noise_out_rej))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_hfsep_out_rej' % (idx + 1), normalize_z(epoched_intrplt_filt_over_100_kx_hfsep_out_rej))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_noise_out_rej' % (idx + 1), normalize_z(epoched_intrplt_filt_over_100_kx_noise_out_rej))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_hfsep_out_rej' % (idx + 1), normalize_z(epoched_intrplt_filt_over_400_kx_hfsep_out_rej))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_noise_out_rej' % (idx + 1), normalize_z(epoched_intrplt_filt_over_400_kx_noise_out_rej))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_hfsep_out_rej' % (idx + 1), normalize_z(epoched_intrplt_filt_500_900_kx_hfsep_out_rej))
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/z_normalized_outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_noise_out_rej' % (idx + 1), normalize_z(epoched_intrplt_filt_500_900_kx_noise_out_rej))

  print('For k00%d z-normalization of basic preprocessing with outlier-rejection made' % (idx + 1))

  ### ### ### ### ###
  ### Advanced Preprocessing on data with outlier-rejection
  ### ### ### ### ###

  # CSP is the signal decomposition using two different signal modalities, due to different points in time, but same preprocessing
  print('Shape of epoched_kx_data_combined_hfsep:')
  csp_epoched_intrplt_kx_data_combined_hfsep_filters_out_rej, csp_epoched_intrplt_kx_data_combined_hfsep_eigenvals_out_rej = meet.spatfilt.CSP(epoched_intrplt_kx_data_combined_hfsep_out_rej[:8].mean(2), epoched_intrplt_kx_data_combined_noise_out_rej[:8].mean(2))
  csp_epoched_intrplt_filt_under_100_kx_filters_out_rej, csp_epoched_intrplt_filt_under_100_kx_eigenvals_out_rej = meet.spatfilt.CSP(epoched_intrplt_filt_under_100_kx_hfsep_out_rej[:8].mean(2), epoched_intrplt_filt_under_100_kx_noise_out_rej[:8].mean(2))
  csp_epoched_intrplt_filt_over_100_kx_filters_out_rej, csp_epoched_intrplt_filt_over_100_kx_eigenvals_out_rej = meet.spatfilt.CSP(epoched_intrplt_filt_over_100_kx_hfsep_out_rej[:8].mean(2), epoched_intrplt_filt_over_100_kx_noise_out_rej[:8].mean(2))
  csp_epoched_intrplt_filt_over_400_kx_filters_out_rej, csp_epoched_intrplt_filt_over_400_kx_eigenvals_out_rej = meet.spatfilt.CSP(epoched_intrplt_filt_over_400_kx_hfsep_out_rej[:8].mean(2), epoched_intrplt_filt_over_400_kx_noise_out_rej[:8].mean(2))
  csp_epoched_intrplt_filt_500_900_kx_filters_out_rej, csp_epoched_intrplt_filt_500_900_kx_eigenvals_out_rej = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep_out_rej[:8].mean(2), epoched_intrplt_filt_500_900_kx_noise_out_rej[:8].mean(2))

  csp_filt_epoched_intrplt_kx_data_combined_out_rej = np.tensordot(csp_epoched_intrplt_kx_data_combined_hfsep_filters_out_rej[:,0].T, kx_data_intrplt_outliers_rejected[:8], axes=(0, 0))
  csp_filt_epoched_intrplt_filt_under_100_kx_out_rej = np.tensordot(csp_epoched_intrplt_filt_under_100_kx_filters_out_rej[:,0].T, intrplt_filt_under_100_kx_out_rej[:8], axes=(0, 0))
  csp_filt_epoched_intrplt_filt_over_100_kx_out_rej = np.tensordot(csp_epoched_intrplt_filt_over_100_kx_filters_out_rej[:,0].T, intrplt_filt_over_100_kx_out_rej[:8], axes=(0, 0))
  csp_filt_epoched_intrplt_filt_over_400_kx_out_rej = np.tensordot(csp_epoched_intrplt_filt_over_400_kx_filters_out_rej[:,0].T, intrplt_filt_over_400_kx_out_rej[:8], axes=(0, 0))
  csp_filt_epoched_intrplt_filt_500_900_kx_out_rej = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters_out_rej[:,0].T, intrplt_filt_500_900_kx_out_rej[:8], axes=(0, 0))

  csp_filt_epoched_intrplt_kx_data_combined_hfsep_out_rej = meet.epochEEG(csp_filt_epoched_intrplt_kx_data_combined_out_rej, triggers_identified_after_rejection, hfSEP_win)
  csp_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej = meet.epochEEG(csp_filt_epoched_intrplt_filt_under_100_kx_out_rej, triggers_identified_after_rejection_under_100_kx_out_rej, hfSEP_win)
  csp_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej = meet.epochEEG(csp_filt_epoched_intrplt_filt_over_100_kx_out_rej, triggers_identified_after_rejection_over_100_kx_out_rej, hfSEP_win)
  csp_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej = meet.epochEEG(csp_filt_epoched_intrplt_filt_over_400_kx_out_rej, triggers_identified_after_rejection_over_400_kx_out_rej, hfSEP_win)
  csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej = meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx_out_rej, triggers_identified_after_rejection_500_900_kx_out_rej, hfSEP_win)

  csp_filt_epoched_intrplt_kx_data_combined_noise_out_rej = meet.epochEEG(csp_filt_epoched_intrplt_kx_data_combined_out_rej, get_indices_for_noise(triggers_identified_after_rejection), noise_win)
  csp_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej = meet.epochEEG(csp_filt_epoched_intrplt_filt_under_100_kx_out_rej, get_indices_for_noise(triggers_identified_after_rejection_under_100_kx_out_rej), noise_win)
  csp_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej = meet.epochEEG(csp_filt_epoched_intrplt_filt_over_100_kx_out_rej, get_indices_for_noise(triggers_identified_after_rejection_over_100_kx_out_rej), noise_win)
  csp_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej = meet.epochEEG(csp_filt_epoched_intrplt_filt_over_400_kx_out_rej, get_indices_for_noise(triggers_identified_after_rejection_over_400_kx_out_rej), noise_win)
  csp_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej = meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx_out_rej, get_indices_for_noise(triggers_identified_after_rejection_500_900_kx_out_rej), noise_win)

  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_kx_data_combined_hfsep_out_rej' % (idx + 1), csp_filt_epoched_intrplt_kx_data_combined_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej' % (idx + 1), csp_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej' % (idx + 1), csp_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej' % (idx + 1), csp_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej' % (idx + 1), csp_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej)

  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_kx_data_combined_noise_out_rej' % (idx + 1), csp_filt_epoched_intrplt_kx_data_combined_noise_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej' % (idx + 1), csp_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej' % (idx + 1), csp_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej' % (idx + 1), csp_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej' % (idx + 1), csp_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej)

  print('For k00%d CSP with outlier-rejection made' % (idx + 1))
  # SSD is the signal decomposition using two different frequency-filtered datasets as different conditions

  print('Skipping SSD for now')
#####  ssd_intrplt_filt_under_100_kx_out_rej_filters, ssd_intrplt_filt_under_100_kx_out_rej_eigenvals = meet.spatfilt.CSP(intrplt_filt_under_100_kx_out_rej[:8], kx_data_intrplt_outliers_rejected[:8])
#####  ssd_intrplt_filt_over_100_kx_out_rej_filters, ssd_intrplt_filt_over_100_kx_out_rej_eigenvals = meet.spatfilt.CSP(intrplt_filt_over_100_kx_out_rej[:8], kx_data_intrplt_outliers_rejected[:8])
#####  ssd_intrplt_filt_over_400_kx_out_rej_filters, ssd_intrplt_filt_over_400_kx_out_rej_eigenvals = meet.spatfilt.CSP(intrplt_filt_over_400_kx_out_rej[:8], kx_data_intrplt_outliers_rejected[:8])
#####  ssd_intrplt_filt_500_900_kx_out_rej_filters, ssd_intrplt_filt_500_900_kx_out_rej_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx_out_rej[:8], kx_data_intrplt_outliers_rejected[:8])
#####
#####  ssd_filt_intrplt_filt_under_100_kx_out_rej = ssd_intrplt_filt_under_100_kx_out_rej_filters.T.dot(intrplt_filt_under_100_kx_out_rej[:8])
#####  ssd_filt_intrplt_filt_over_100_kx_out_rej = ssd_intrplt_filt_over_100_kx_out_rej_filters.T.dot(intrplt_filt_over_100_kx_out_rej[:8])
#####  ssd_filt_intrplt_filt_over_400_kx_kx_out_rej = ssd_intrplt_filt_over_400_kx_out_rej_filters.T.dot(intrplt_filt_over_400_kx_out_rej[:8])
#####  ssd_filt_intrplt_filt_500_900_kx_out_rej = ssd_intrplt_filt_500_900_kx_out_rej_filters.T.dot(intrplt_filt_500_900_kx_out_rej[:8])
#####
#####  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_under_100_kx_out_rej' % (idx + 1), ssd_filt_intrplt_filt_under_100_kx_out_rej)
#####  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_over_100_kx_out_rej' % (idx + 1), ssd_filt_intrplt_filt_over_100_kx_out_rej)
#####  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_over_400_kx_kx_out_rej' % (idx + 1), ssd_filt_intrplt_filt_over_400_kx_kx_out_rej)
#####  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_500_900_kx_out_rej' % (idx + 1), ssd_filt_intrplt_filt_500_900_kx_out_rej)

  print('For k00%d SSD with outlier-rejection made' % (idx + 1))
  # CCAr is the technique that tries to derive filters that 'modify' the single-trial to be more similar to the single-trial averages
  a_epoched_intrplt_kx_data_combined_hfsep_out_rej, b_epoched_intrplt_kx_data_combined_hfsep_out_rej, s_epoched_intrplt_kx_data_combined_hfsep_out_rej = meet.spatfilt.CCAvReg(epoched_intrplt_kx_data_combined_hfsep_out_rej[:8,:,:])
  a_epoched_intrplt_filt_under_100_kx_hfsep_out_rej, b_epoched_intrplt_filt_under_100_kx_hfsep_out_rej, s_epoched_intrplt_filt_under_100_kx_hfsep_out_rej = meet.spatfilt.CCAvReg(epoched_intrplt_filt_under_100_kx_hfsep_out_rej[:8,:,:])
  a_epoched_intrplt_filt_over_100_kx_hfsep_out_rej, b_epoched_intrplt_filt_over_100_kx_hfsep_out_rej, s_epoched_intrplt_filt_over_100_kx_hfsep_out_rej = meet.spatfilt.CCAvReg(epoched_intrplt_filt_over_100_kx_hfsep_out_rej[:8,:,:])
  a_epoched_intrplt_filt_over_400_kx_hfsep_out_rej, b_epoched_intrplt_filt_over_400_kx_hfsep_out_rej, s_epoched_intrplt_filt_over_400_kx_hfsep_out_rej = meet.spatfilt.CCAvReg(epoched_intrplt_filt_over_400_kx_hfsep_out_rej[:8,:,:])
  a_epoched_intrplt_filt_500_900_kx_hfsep_out_rej, b_epoched_intrplt_filt_500_900_kx_hfsep_out_rej, s_epoched_intrplt_filt_500_900_kx_hfsep_out_rej = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep_out_rej[:8,:,:])

  ccar_filt_epoched_intrplt_kx_data_combined_hfsep_out_rej = np.tensordot(a_epoched_intrplt_kx_data_combined_hfsep_out_rej[:,0], epoched_intrplt_kx_data_combined_hfsep_out_rej[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_kx_data_combined_noise_out_rej = np.tensordot(a_epoched_intrplt_kx_data_combined_hfsep_out_rej[:,0], epoched_intrplt_kx_data_combined_noise_out_rej[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej = np.tensordot(a_epoched_intrplt_filt_under_100_kx_hfsep_out_rej[:,0], epoched_intrplt_filt_under_100_kx_hfsep_out_rej[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej = np.tensordot(a_epoched_intrplt_filt_under_100_kx_hfsep_out_rej[:,0], epoched_intrplt_filt_under_100_kx_noise_out_rej[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej = np.tensordot(a_epoched_intrplt_filt_over_100_kx_hfsep_out_rej[:,0], epoched_intrplt_filt_over_100_kx_hfsep_out_rej[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej = np.tensordot(a_epoched_intrplt_filt_over_100_kx_hfsep_out_rej[:,0], epoched_intrplt_filt_over_100_kx_noise_out_rej[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej = np.tensordot(a_epoched_intrplt_filt_over_400_kx_hfsep_out_rej[:,0], epoched_intrplt_filt_over_400_kx_hfsep_out_rej[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej = np.tensordot(a_epoched_intrplt_filt_over_400_kx_hfsep_out_rej[:,0], epoched_intrplt_filt_over_400_kx_noise_out_rej[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep_out_rej[:,0], epoched_intrplt_filt_500_900_kx_hfsep_out_rej[:8,:,:], axes=(0, 0))
  ccar_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep_out_rej[:,0], epoched_intrplt_filt_500_900_kx_noise_out_rej[:8,:,:], axes=(0, 0))

  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_kx_data_combined_hfsep_out_rej' % (idx + 1), ccar_filt_epoched_intrplt_kx_data_combined_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_kx_data_combined_noise_out_rej' % (idx + 1), ccar_filt_epoched_intrplt_kx_data_combined_noise_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej' % (idx + 1), ccar_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej' % (idx + 1), ccar_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej' % (idx + 1), ccar_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej' % (idx + 1), ccar_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej' % (idx + 1), ccar_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej' % (idx + 1), ccar_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej' % (idx + 1), ccar_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej' % (idx + 1), ccar_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej)

  print('For k00%d CCAr with outlier-rejection made' % (idx + 1))
  # bCSPT is the technique that tries to derive filters in the spatial and in the temporal domain, leading to the use of convolution
  s_outepoched_intrplt_filt_under_100_kx_hfsep_eigenvals_out_rej, t_outepoched_intrplt_filt_under_100_kx_hfsep_eigenvals_out_rej, W_out_epoched_intrplt_filt_under_100_kx_hfsep_out_rej, V_out_epoched_intrplt_filt_under_100_kx_hfsep_out_rej = bCSTP(epoched_intrplt_filt_under_100_kx_hfsep_out_rej[:8,:,:], epoched_intrplt_filt_under_100_kx_noise_out_rej[:8,:,:], num_iter=15, t_keep=2, s_keep=2)
  s_outepoched_intrplt_filt_over_100_kx_hfsep_eigenvals_out_rej, t_outepoched_intrplt_filt_over_100_kx_hfsep_eigenvals_out_rej, W_out_epoched_intrplt_filt_over_100_kx_hfsep_out_rej, V_out_epoched_intrplt_filt_over_100_kx_hfsep_out_rej = bCSTP(epoched_intrplt_filt_over_100_kx_hfsep_out_rej[:8,:,:], epoched_intrplt_filt_over_100_kx_noise_out_rej[:8,:,:], num_iter=15, t_keep=2, s_keep=2)
  s_outepoched_intrplt_filt_over_400_kx_hfsep_eigenvals_out_rej, t_outepoched_intrplt_filt_over_400_kx_hfsep_eigenvals_out_rej, W_out_epoched_intrplt_filt_over_400_kx_hfsep_out_rej, V_out_epoched_intrplt_filt_over_400_kx_hfsep_out_rej = bCSTP(epoched_intrplt_filt_over_400_kx_hfsep_out_rej[:8,:,:], epoched_intrplt_filt_over_400_kx_noise_out_rej[:8,:,:], num_iter=15, t_keep=2, s_keep=2)
  s_outepoched_intrplt_filt_500_900_kx_hfsep_eigenvals_out_rej, t_outepoched_intrplt_filt_500_900_kx_hfsep_eigenvals_out_rej, W_out_epoched_intrplt_filt_500_900_kx_hfsep_out_rej, V_out_epoched_intrplt_filt_500_900_kx_hfsep_out_rej = bCSTP(epoched_intrplt_filt_500_900_kx_hfsep_out_rej[:8,:,:], epoched_intrplt_filt_500_900_kx_noise_out_rej[:8,:,:], num_iter=15, t_keep=2, s_keep=2)

  bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_out_rej = scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_filt_under_100_kx_hfsep_out_rej[-1][:,0], intrplt_filt_under_100_kx_out_rej[:8]), V_out_epoched_intrplt_filt_under_100_kx_hfsep_out_rej[-1][:,0][::-1], axis=-1)
  bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_out_rej = scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_filt_over_100_kx_hfsep_out_rej[-1][:,0], intrplt_filt_over_100_kx_out_rej[:8]), V_out_epoched_intrplt_filt_over_100_kx_hfsep_out_rej[-1][:,0][::-1], axis=-1)
  bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_out_rej = scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_filt_over_400_kx_hfsep_out_rej[-1][:,0], intrplt_filt_over_400_kx_out_rej[:8]), V_out_epoched_intrplt_filt_over_400_kx_hfsep_out_rej[-1][:,0][::-1], axis=-1)
  bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_out_rej = scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_filt_500_900_kx_hfsep_out_rej[-1][:,0], intrplt_filt_500_900_kx_out_rej[:8]), V_out_epoched_intrplt_filt_500_900_kx_hfsep_out_rej[-1][:,0][::-1], axis=-1)

  bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_out_rej, triggers_identified_after_rejection_under_100_kx_out_rej, hfSEP_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_out_rej, get_indices_for_noise(triggers_identified_after_rejection_under_100_kx_out_rej), noise_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_out_rej, triggers_identified_after_rejection_over_100_kx_out_rej, hfSEP_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_out_rej, get_indices_for_noise(triggers_identified_after_rejection_over_100_kx_out_rej), noise_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_out_rej, triggers_identified_after_rejection_over_400_kx_out_rej, hfSEP_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_out_rej, get_indices_for_noise(triggers_identified_after_rejection_over_400_kx_out_rej), noise_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_out_rej, triggers_identified_after_rejection_500_900_kx_out_rej, hfSEP_win)
  bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej = meet.epochEEG(bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_out_rej, get_indices_for_noise(triggers_identified_after_rejection_500_900_kx_out_rej), noise_win)

  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_under_100_kx_noise_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_over_100_kx_noise_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_over_400_kx_noise_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_hfsep_out_rej)
  save('/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej' % (idx + 1), bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx_noise_out_rej)

  print('For k00%d bCSTP with outlier-rejection made' % (idx + 1))

### ### ### ### ###
### Finally call the two prep-methods individually, to save on memory
### ### ### ### ###
prep_without_outlier_rejection()
prep_with_outlier_rejection()

### ### ### ### ###
### End of Data Preprocessing Script
### ### ### ### ###