triggers_k4_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/kx_data_combined.npy' % 4, allow_pickle=True)                                                                  
data_k4_combined = triggers_k4_combined                                                                                                                                                                   
triggers_k4_combined = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/triggers_for_kx_combined.npy' % 4, allow_pickle=True) 

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
hfsep_around_artifact = meet.epochEEG(data_k4_combined, triggers_k4_combined, [-497, 503])
std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) + std_basic_prep, (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\nstimulus-artifact at 0ms, hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


### Plots per spectral filter modality
intrplt_filt_under_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_under_100_kx.npy' % 4, allow_pickle=True)
hfsep_around_artifact = meet.epochEEG(intrplt_filt_under_100_kx, triggers_k4_combined, [-500, 500])
std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) + std_basic_prep, (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered under 100Hz, hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


intrplt_filt_over_100_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_100_kx.npy' % 4, allow_pickle=True)
hfsep_around_artifact = meet.epochEEG(intrplt_filt_over_100_kx, triggers_k4_combined, [-500, 500])
std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) + std_basic_prep, (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered above 100Hz, hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


intrplt_filt_over_400_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_400_kx.npy' % 4, allow_pickle=True)
hfsep_around_artifact = meet.epochEEG(intrplt_filt_over_400_kx, triggers_k4_combined, [-500, 500])
std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) + std_basic_prep, (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered above 400Hz, hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


intrplt_filt_500_900_kx = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy' % 4, allow_pickle=True)
hfsep_around_artifact = meet.epochEEG(intrplt_filt_500_900_kx, triggers_k4_combined, [-500, 500])
std_basic_prep = np.std(hfsep_around_artifact[0,:,:]-hfsep_around_artifact[5,:,:], axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) + std_basic_prep, (hfsep_around_artifact[0,:,3]-hfsep_around_artifact[5,:,3]) - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('voltage in μV', fontproperties=font)
plt.title('bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
hfsep_around_artifact = meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k4_combined, [-500, 500])
std_basic_prep = np.std(hfsep_around_artifact, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact[:,3] + std_basic_prep, hfsep_around_artifact[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('CCAr-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


# ToDo: Do the averaging and the plot-creation after averaging now!
epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k4_combined, [-500, 500]), weights=np.ones(10) / 10, axis=1)[:,int(10 / 2):-int(10 / 2):10]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 10 samples CCAr-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k4_combined, [-500, 500]), weights=np.ones(20) / 20, axis=1)[:,int(20 / 2):-int(20 / 2):20]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 20 samples CCAr-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ccar_filt_epoched_intrplt_filt_500_900_kx, triggers_k4_combined, [-500, 500]), weights=np.ones(500) / 500, axis=1)[:,int(500 / 2):-int(500 / 2):500]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 500 samples CCAr-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
hfsep_around_artifact = meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k4_combined, [-500, 500])
std_basic_prep = np.std(hfsep_around_artifact, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact[:,3] + std_basic_prep, hfsep_around_artifact[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('CSP-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k4_combined, [-500, 500]), weights=np.ones(10) / 10, axis=1)[:,int(10 / 2):-int(10 / 2):10]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 10 samples CSP-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k4_combined, [-500, 500]), weights=np.ones(20) / 20, axis=1)[:,int(20 / 2):-int(20 / 2):20]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 20 samples CSP-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(csp_filt_epoched_intrplt_filt_500_900_kx, triggers_k4_combined, [-500, 500]), weights=np.ones(500) / 500, axis=1)[:,int(500 / 2):-int(500 / 2):500]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 500 samples CSP-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


#SSD
epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx[:8], data_k4_combined[:8])
ssd_filt_intrplt_filt_500_900_kx = np.tensordot(ssd_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
# hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ssd_filt_intrplt_filt_500_900_kx, triggers_k4_combined, [-500, 500]), weights=np.ones(500) / 500, axis=1)[:,int(500 / 2):-int(500 / 2):500]
std_basic_prep = np.std(meet.epochEEG(ssd_filt_intrplt_filt_500_900_kx, triggers_k4_combined, [-500, 500]), axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('SSD-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx[:8], data_k4_combined[:8])
ssd_filt_intrplt_filt_500_900_kx = np.tensordot(ssd_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ssd_filt_intrplt_filt_500_900_kx, triggers_k4_combined, [-500, 500]), weights=np.ones(10) / 10, axis=1)[:,int(10 / 2):-int(10 / 2):10]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 10 samples SSD-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx[:8], data_k4_combined[:8])
ssd_filt_intrplt_filt_500_900_kx = np.tensordot(ssd_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
hfsep_around_artifact_mean_ten = convolve1d(meet.epochEEG(ssd_filt_intrplt_filt_500_900_kx, triggers_k4_combined, [-500, 500]), weights=np.ones(500) / 500, axis=1)[:,int(500 / 2):-int(500 / 2):500]
std_basic_prep = np.std(hfsep_around_artifact_mean_ten, axis=1)
plt.plot(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3], linewidth=2, label='single-trial example', color='red')
plt.fill_between(np.arange(-500, 500), hfsep_around_artifact_mean_ten[:,3] + std_basic_prep, hfsep_around_artifact_mean_ten[:,3] - std_basic_prep, color='blue', label='+/- STD across single-trials', alpha=0.3)
plt.xticks(ticks=ticks, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus', fontproperties=font)
plt.ylabel('relative feature value in μV', fontproperties=font)
plt.title('Sub-Average w. 500 samples SSD-spatially weighted hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20, loc=2)












TODO:
################
################

from pyts.image import RecurrencePlot

rp = RecurrencePlot()
# rp.fit_transform(np.swapaxes(x_dat, 0, 1))
ticks = np.arange(0, 1100, 100)
ticklabels = ['%d' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((data_k4_combined[0,triggers_k4_combined[3]-497:triggers_k4_combined[3]+503]-data_k4_combined[5,triggers_k4_combined[3]-497:triggers_k4_combined[3]+503]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.title('recurrence-plot of bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\nstimulus-artifact at 0ms, hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)


ticks = np.arange(0, 1100, 100)
ticklabels = ['%d' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((intrplt_filt_under_100_kx[0,triggers_k4_combined[3]-500:triggers_k4_combined[3]+500]-intrplt_filt_under_100_kx[5,triggers_k4_combined[3]-500:triggers_k4_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.title('recurrence-plot of bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered under 100Hz, hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)


ticks = np.arange(0, 1100, 100)
ticklabels = ['%d' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((intrplt_filt_over_100_kx[0,triggers_k4_combined[3]-500:triggers_k4_combined[3]+500]-intrplt_filt_over_100_kx[5,triggers_k4_combined[3]-500:triggers_k4_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.title('recurrence-plot of bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered over 100Hz, hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)


ticks = np.arange(0, 1100, 100)
ticklabels = ['%d' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((intrplt_filt_over_400_kx[0,triggers_k4_combined[3]-500:triggers_k4_combined[3]+500]-intrplt_filt_over_400_kx[5,triggers_k4_combined[3]-500:triggers_k4_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.title('recurrence-plot of bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered over 400Hz, hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)


ticks = np.arange(0, 1100, 100)
ticklabels = ['%d' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((intrplt_filt_500_900_kx[0,triggers_k4_combined[3]-500:triggers_k4_combined[3]+500]-intrplt_filt_500_900_kx[5,triggers_k4_combined[3]-500:triggers_k4_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.title('recurrence-plot of bipolar montage FZ-CP5 showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
a_epoched_intrplt_filt_500_900_kx_hfsep, b_epoched_intrplt_filt_500_900_kx_hfsep, s_epoched_intrplt_filt_500_900_kx_hfsep = meet.spatfilt.CCAvReg(epoched_intrplt_filt_500_900_kx_hfsep[:8,:,:])
ccar_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(a_epoched_intrplt_filt_500_900_kx_hfsep[:,0], intrplt_filt_500_900_kx[:8,:], axes=(0, 0))
ticks = np.arange(0, 1100, 100)
ticklabels = ['%d' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((csp_filt_epoched_intrplt_filt_500_900_kx[triggers_k4_combined[3]-500:triggers_k4_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.title('recurrence-plot of CSP-filtered data showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
csp_epoched_intrplt_filt_500_900_kx_filters, csp_epoched_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(epoched_intrplt_filt_500_900_kx_hfsep[:8].reshape(8, -1), epoched_intrplt_filt_500_900_kx_noise[:8].reshape(8, -1))
csp_filt_epoched_intrplt_filt_500_900_kx = np.tensordot(csp_epoched_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
ticks = np.arange(0, 1100, 100)
ticklabels = ['%d' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((ccar_filt_epoched_intrplt_filt_500_900_kx[triggers_k4_combined[3]-500:triggers_k4_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.title('recurrence-plot of CCAr filtered data showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)


epoched_intrplt_filt_500_900_kx_hfsep = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % 4, allow_pickle=True)
epoched_intrplt_filt_500_900_kx_noise = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % 4, allow_pickle=True)
ssd_intrplt_filt_500_900_kx_filters, ssd_intrplt_filt_500_900_kx_eigenvals = meet.spatfilt.CSP(intrplt_filt_500_900_kx[:8], data_k4_combined[:8])
ssd_filt_intrplt_filt_500_900_kx = np.tensordot(ssd_intrplt_filt_500_900_kx_filters[:,0].T, intrplt_filt_500_900_kx[:8], axes=(0, 0))
ticks = np.arange(0, 1100, 100)
ticklabels = ['%d' % tick for tick in ((ticks - 500) / 10)]
plt.imshow(np.squeeze(rp.fit_transform((ssd_filt_intrplt_filt_500_900_kx[triggers_k4_combined[3]-500:triggers_k4_combined[3]+500]).reshape(1, -1)), axis=0))
plt.colorbar()
plt.xticks(ticks=ticks, labels=ticklabels)
plt.yticks(ticks=ticks.T, labels=ticklabels)
plt.xlabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.ylabel('time in single-trial relative to stimulus in ms', fontproperties=font)
plt.title('recurrence-plot of SSD-filtered data showing hfSEP example in surrounding noise\ndata interpolated and IIR-filtered [500Hz - 900Hz], hfSEP between ~15ms and ~30ms\ndata from subject with lowest SNNR in single-trials', fontproperties=font)
plt.tick_params(labelsize=20)
