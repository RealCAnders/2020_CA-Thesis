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

ToDo: Add the things from the beginning

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

triggers_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/kx_data_combined.npy' % 3, allow_pickle=True)                                                                  
data_k3_combined = triggers_k3_combined                                                                                                                                                                   
triggers_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined.npy' % 3, allow_pickle=True) 

font = FontProperties()  
font.set_family('serif')  
font.set_name('Times New Roman')  
font.set_size(20)        
ticks = np.arange(-500, 600, 100)                                                                                                                                                                        
ticklabels = ['%d ms' % tick for tick in (ticks / 10)]


#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
# basic plot
hfsep_around_artifact = meet.epochEEG(data_k3_combined, triggers_k3_combined, [-497, 503])
std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) + std_basic_prep, (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\nstimulus-artifact at 0ms, hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


### Plots per spectral filter modality
intrplt_filt_under_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_under_100_kx.npy' % 3, allow_pickle=True)
hfsep_around_artifact = meet.epochEEG(intrplt_filt_under_100_kx, triggers_k3_combined, [-500, 500])
std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) + std_basic_prep, (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered under 100Hz, hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


intrplt_filt_over_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_100_kx.npy' % 3, allow_pickle=True)
hfsep_around_artifact = meet.epochEEG(intrplt_filt_over_100_kx, triggers_k3_combined, [-500, 500])
std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) + std_basic_prep, (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered above 100Hz, hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


intrplt_filt_over_400_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_400_kx.npy' % 3, allow_pickle=True)
hfsep_around_artifact = meet.epochEEG(intrplt_filt_over_400_kx, triggers_k3_combined, [-500, 500])
std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) + std_basic_prep, (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered above 400Hz, hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


intrplt_filt_500_900_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy' % 3, allow_pickle=True)
hfsep_around_artifact = meet.epochEEG(intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500])
std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) + std_basic_prep, (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 3, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 3, allow_pickle=True)
a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
hfsep_around_artifact = meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500])
std_basic_prep = np.std(hfsep_around_artifact, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact[:,3] + std_basic_prep, hfsep_around_artifact[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('CCAr-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


# ToDo: Do the averaging and the plot-creation after averaging now!
epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 3, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 3, allow_pickle=True)
a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(10) / 10, axis=1)[:,int(10 / 2):-int(10 / 2):10]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 10 samples CCAr-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 3, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 3, allow_pickle=True)
a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(20) / 20, axis=1)[:,int(20 / 2):-int(20 / 2):20]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 20 samples CCAr-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 3, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 3, allow_pickle=True)
a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(500) / 500, axis=1)[:,int(500 / 2):-int(500 / 2):500]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 500 samples CCAr-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 3, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 3, allow_pickle=True)
csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
hfsep_around_artifact = meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500])
std_basic_prep = np.std(hfsep_around_artifact, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact[:,3] + std_basic_prep, hfsep_around_artifact[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('CSP-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 3, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 3, allow_pickle=True)
csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(10) / 10, axis=1)[:,int(10 / 2):-int(10 / 2):10]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 10 samples CSP-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 3, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 3, allow_pickle=True)
csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(20) / 20, axis=1)[:,int(20 / 2):-int(20 / 2):20]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 20 samples CSP-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 3, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 3, allow_pickle=True)
csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(500) / 500, axis=1)[:,int(500 / 2):-int(500 / 2):500]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 500 samples CSP-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


#SSD
epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 3, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 3, allow_pickle=True)
ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx[:8], data_k3_combined[:8])
ssd_filt_intrplt_filt_500_900_kx = np.tensordot(ssd_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
# hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ssd_filt_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(500) / 500, axis=1)[:,int(500 / 2):-int(500 / 2):500]
std_basic_prep = np.std(meet.epochEEG(ssd_filt_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('SSD-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 3, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 3, allow_pickle=True)
ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx[:8], data_k3_combined[:8])
ssd_filt_intrplt_filt_500_900_kx = np.tensordot(ssd_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ssd_filt_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(10) / 10, axis=1)[:,int(10 / 2):-int(10 / 2):10]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 10 samples SSD-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 3, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 3, allow_pickle=True)
ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx[:8], data_k3_combined[:8])
ssd_filt_intrplt_filt_500_900_kx = np.tensordot(ssd_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ssd_filt_intrplt_filt_500_900_kx, triggers_k3_combined, [-500, 500]), weights=np.ones(500) / 500, axis=1)[:,int(500 / 2):-int(500 / 2):500]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 500 samples SSD-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


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
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus', fontproperties=font)
plt.title('recurrence-plot of bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\nstimulus-artifact at 0ms, hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


ticks = np.arange(0, 1100, 100)
ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((intrplt_filt_under_100_kx[0,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]-intrplt_filt_under_100_kx[5,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus', fontproperties=font)
plt.title('recurrence-plot of bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered under 100Hz, hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


ticks = np.arange(0, 1100, 100)
ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((intrplt_filt_over_100_kx[0,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]-intrplt_filt_over_100_kx[5,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus', fontproperties=font)
plt.title('recurrence-plot of bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered over 100Hz, hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


ticks = np.arange(0, 1100, 100)
ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((intrplt_filt_over_400_kx[0,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]-intrplt_filt_over_400_kx[5,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus', fontproperties=font)
plt.title('recurrence-plot of bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered over 400Hz, hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


ticks = np.arange(0, 1100, 100)
ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((intrplt_filt_500_900_kx[0,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]-intrplt_filt_500_900_kx[5,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus', fontproperties=font)
plt.title('recurrence-plot of bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


ticks = np.arange(0, 1100, 100)
ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((csp_filt_epoched_intrplt_filt_500_900_kx[triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus', fontproperties=font)
plt.title('recurrence-plot of CSP-filtered data showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


ticks = np.arange(0, 1100, 100)
ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((ccar_filt_epoched_intrplt_filt_500_900_kx[triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus', fontproperties=font)
plt.title('recurrence-plot of CCAr filtered data showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


ticks = np.arange(0, 1100, 100)
ticklabels = ['%d ms' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((ssd_filt_intrplt_filt_500_900_kx[triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus', fontproperties=font)
plt.title('recurrence-plot of SSD-filtered data showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)




#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
# plot after refined with med. experts:
intrplt_filt_500_900_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy' % 3, allow_pickle=True)
plt.plot(np.arange(-500, 500), intrplt_filt_500_900_kx[0,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500]-intrplt_filt_500_900_kx[5,triggers_k3_combined[3]-500:triggers_k3_combined[3]+500])
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 3, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 3, allow_pickle=True)
csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
plt.plot(np.arange(-500, 500), csp_filt_epoched_intrplt_filt_500_900_kx[triggers_k3_combined[3]-500:triggers_k3_combined[3]+500])
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('CSP-filtered data showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
plt.plot(np.arange(-500, 500), ccar_filt_epoched_intrplt_filt_500_900_kx[triggers_k3_combined[3]-500:triggers_k3_combined[3]+500])
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('CCAr-filtered data showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
# ToDo: BCSTP läuft nicht wirklich... Mit Gunnar besprechen!
s_outepoched_intrplt_filt_500_900_kx_hfsep_eigenvals, t_outepoched_intrplt_filt_500_900_kx_hfsep_eigenvals, W_out_epoched_intrplt_filt_500_900_kx_hfsep, V_out_epoched_intrplt_filt_500_900_kx_hfsep = bCSTP(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:], epoched_intrplt_filt_500_900_kx_noise[:8,:,:], num_iter=15, t_keep=2, s_keep=2)

[::-1]

bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx = scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_filt_500_900_kx_hfsep[-1][:,0], intrplt_filt_500_900_kx[:8]), V_out_epoched_intrplt_filt_500_900_kx_hfsep[-1][:,0], axis=-1)
plt.plot(np.arange(-500, 500), bcstp_spat_temp_filt_epoched_intrplt_filt_500_900_kx[triggers_k3_combined[3]-500:triggers_k3_combined[3]+500])
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative filter-activation', fontproperties=font)
plt.title('bCSTP-filtered data showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)




#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
epoched_intrplt_kx_data_combined_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy' % 3, allow_pickle=True)
epoched_intrplt_kx_data_combined_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy' % 3, allow_pickle=True)
ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_kx_data_combined_hfsep[:8].reshape(8, -1))
ssd_filt_intrplt_filt_500_900_kx = np.tensordot(ssd_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
plt.plot(np.arange(-500, 500), ssd_filt_intrplt_filt_500_900_kx[triggers_k3_combined[3]-500:triggers_k3_combined[3]+500])
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('SSD-filtered data showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with highest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)



#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
# ED-Plots for Outlier-Rejection zoomed out


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
                                 

triggers_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/kx_data_combined.npy' % 3, allow_pickle=True)                           
                                 
data_k3_combined = triggers_k3_combined                                                                                                                            
                                 
triggers_k3_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined.npy' % 3, allow_pickle=True)                   
lsd_baseline = []  
euclidean_distances = []  
srate = 10000                                                                                                                                                      
num_secs = int(len(data_k3_combined[0]) / srate)
chan = 0   

pxx_per_sec = np.asarray([scipy.signal.welch(data_k3_combined[chan, srate * i : srate * (i + 1)], fs=srate)[1] for i in range(num_secs)])   
lsd_baseline = np.median(pxx_per_sec, axis=0).mean(0)  
euclidean_distances = np.sqrt(((lsd_baseline - pxx_per_sec)**2).sum(1))                                                                                    

font = FontProperties()   
font.set_family('serif')   
font.set_name('Times New Roman')   
font.set_size(20)       

plt.plot(euclidean_distances, color='red', label='ED in channel FZ')  
plt.plot(np.ones(len(euclidean_distances)) * (0.2 * 3), color='blue', linestyle=':', label='Threshold %.2f, 0.33%% rejection' % (0.2 * 3), linewidth=2)  
plt.plot(np.ones(len(euclidean_distances)) * (0.1 * 3), color='#000000', linestyle='--', label='Threshold %.2f, 4.84%% rejection' % (0.1 * 3), linewidth=2)
plt.xlim([0, len(euclidean_distances)])  
plt.ylim([0, 1])
plt.title('Plot of euclidean distances (ED) between single-trial spectrum and fit of averaged spectrum; K003', fontproperties=font)  
plt.xlabel('Seconds in combined data of K003', fontproperties=font)  
plt.ylabel('Euclidean Distance', fontproperties=font)  
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)
plt.show()


# zoomed in
plt.plot(euclidean_distances, color='red', label='ED in channel FZ')  
plt.plot(np.ones(len(euclidean_distances)) * (0.25), color='blue', linestyle=':', label='Threshold %.2f, 10%% rejection' % (0.25), linewidth=2)  
plt.plot(np.ones(len(euclidean_distances)) * (0.2), color='#000000', linestyle='--', label='Threshold %.2f, 25%% rejection' % (0.2), linewidth=2)  
plt.xlim([0, len(euclidean_distances)])
plt.title('Plot of euclidean distances (ED) between single-trial spectrum and fit of averaged spectrum; K003', fontproperties=font)  
plt.xlabel('Seconds in combined data of K003', fontproperties=font)  
plt.ylim([0.1, 0.3])
plt.ylabel('Euclidean Distance', fontproperties=font)  
plt.tick_params(labelsize=15)
plt.legend(fontsize=15, loc=2)
plt.show()



##########
# Plots for variability-analysis:
# Desktop/charite_master_thesis/ToDo's/mid_term_plots_code_snippets --> Search for variability across sub-averages

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