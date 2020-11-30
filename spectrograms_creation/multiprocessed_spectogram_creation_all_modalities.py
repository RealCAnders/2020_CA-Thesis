from multiprocessing import Process
import meet
import numpy as np
import scipy
import datetime
import matplotlib.pyplot as plt
from numpy import load

offset = 1000
s_rate = 10000
stim_per_sec = 4

def get_indices_for_noise(triggers_to_get_indices_for):
  """
  General Idea:
  If we defined a window of 'safety', which we have to randomly place the noise-window in,
  then we can safely use random noise-window placements in this window.
  The window around a stimulus that we're interested in is 150ms long: 
  [<<-55ms Noise? -10ms>> <<-8ms Intrplt 2ms>> <<5ms hfSEP 45ms>> <<45ms Noise? 95ms>>]:= 150ms
  --> Follows the noise-window can start in [-55ms to -50ms] and [45ms to 55ms].
  """
  possible_neg_noise_starting_indices = np.arange(-550, -500, 1)
  possible_pos_starting_indices = np.arange(450, 550)
  possible_noise_indices_per_sample = np.append(possible_neg_noise_starting_indices, possible_pos_starting_indices)
  return np.random.choice(possible_noise_indices_per_sample, len(triggers_to_get_indices_for))

# ToDo: Noise-Spectogram Creation
def create_spectograms_and_store_them(conc_data, subject_id, title, lower, upper, indizes):
	print(conc_data.shape)
	for idx in np.arange(1, len(conc_data) - 1, 1): 
		coors, s = meet.tf.gft(conc_data[(idx * 10000) - 550 : ((idx + 1) * 10000) + 550], axis=0, sampling=custom_sampling_meg) 
		t, f, tf_mean_p_interp = meet.tf.interpolate_gft(coors, s, IM_shape=(11100 // 2, 11100), data_len=11100, kindf='nearest', kindt='nearest')  
		mean_power_frequency = tf_mean_p_interp.mean(-1)
		for i in range(4): 
#			plt.pcolormesh(t[(2500*i):(2500*(i + 1)) + 1], f[:1001], np.abs(np.multiply(tf_mean_p_interp[:1000, (2500*i):(2500*(i + 1))], np.expand_dims(mean_power_frequency[:1000], 1)))) 
			plt.pcolormesh(t[(2500*i) + 550 + indizes[(idx * 4) + i]:(2500*(i + 1)) + 1 + 550 + indizes[(idx * 4) + i]], f[:upper + 1], np.abs(tf_mean_p_interp[:upper, (2500*i) + 550 + indizes[(idx * 4) + i]:(2500*(i + 1)) + 550 + indizes[(idx * 4) + i]])) 
			plt.ylim((lower, upper)) 
			plt.colorbar() 
			#plt.savefig('/media/christoph/Volume/Masterthesis/testme/k00%d_%s_%d_%d' % ((subject_id + 1), title, idx, i), dpi=80)
			plt.savefig('/media/christoph/Volume/Masterthesis/spectograms_noise/k00%d/%s_%d_%d' % ((subject_id + 1), title, idx, i), dpi=50) 
			plt.clf() 
			plt.close('all')

# create a custom sampling scheme for the S transform 
def custom_sampling_meg(N): 
    S_frange = [5, 5000] 
    S_fnum = 30 
    S_Nperperiod = 4 
    wanted_freqs = np.exp(np.linspace(np.log(S_frange[0]), 
        np.log(S_frange[1]), S_fnum)) 
    fftfreqs = np.fft.fftfreq(N, d=1./meg_srate) 
    # find the nearest frequency indices 
    y = np.unique([np.argmin((w - fftfreqs)**2) 
        for w in wanted_freqs]) 
    x = ((S_Nperperiod*fftfreqs[y]*N/float(meg_srate))//2).astype(int) 
    return x,y
meg_srate = 10000

data_to_be_analyzed_in_cp5_min_fz = [
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/kx_data_combined.npy', 0, 200],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_kx_data_combined.npy', 0, 200],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_under_100_kx.npy', 0, 200],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_100_kx.npy', 100, 1500],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_over_400_kx.npy', 400, 1500],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/intrplt_filt_500_900_kx.npy', 500, 900]
]

data_to_be_used_as_is = [
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_kx_data_combined.npy', 0, 200],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_intrplt_kx_data_combined.npy', 0, 200],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_intrplt_filt_under_100_kx.npy', 0, 200],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_intrplt_filt_over_100_kx.npy', 100, 1500],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_intrplt_filt_over_400_kx.npy', 400, 1500],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_intrplt_filt_500_900_kx.npy', 500, 900],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_under_100_kx.npy', 0, 200],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_100_kx.npy', 100, 1500],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_400_kx.npy', 400, 1500],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_500_900_kx.npy', 500, 900],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_kx_data_combined.npy', 0, 200],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_intrplt_kx_data_combined.npy', 0, 200],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_intrplt_filt_under_100_kx.npy', 0, 200],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_intrplt_filt_over_100_kx.npy', 100, 1500],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_intrplt_filt_over_400_kx.npy', 400, 1500],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_intrplt_filt_500_900_kx.npy', 500, 900],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_intrplt_kx_data_combined.npy', 0, 200],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_intrplt_filt_under_100_kx.npy', 0, 200],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_intrplt_filt_over_100_kx.npy', 100, 1500],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_intrplt_filt_over_400_kx.npy', 400, 1500],
	['/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_intrplt_filt_500_900_kx.npy', 500, 900]
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_intrplt_kx_data_combined_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_intrplt_filt_under_100_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_intrplt_filt_over_100_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_intrplt_filt_over_400_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_intrplt_filt_500_900_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_under_100_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_over_100_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_over_400_kx_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_500_900_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_intrplt_kx_data_combined_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_intrplt_filt_under_100_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_intrplt_filt_over_100_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_intrplt_filt_over_400_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_intrplt_filt_500_900_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_intrplt_filt_under_100_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_intrplt_filt_over_100_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_intrplt_filt_over_400_kx_out_rej.npy',
#-#	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_intrplt_filt_500_900_kx_out_rej.npy'
]

print('Starting by ' + str(datetime.datetime.now()))

# Done already: [2, 3], [1, 9], [4, 5, 6], [0, 7]
# [1, 2, 3, 4, 0, 5, 6, 7, 8, 9]
#for i in [TODODODO]:
i = 8
print('Now working on subject k00%d' % (i + 1))

processes = []

for thread_id, descriptor in enumerate(data_to_be_analyzed_in_cp5_min_fz):
	file_path, lower, upper = descriptor
	conc_data = load(file_path % (i + 1))
	title = file_path.split('/')[-1].split('.')[0]

	conc_data = conc_data[5] - conc_data[0]
	print('Length of %s is: [%d]' % (title, len(conc_data)))
	indizes = get_indices_for_noise(np.arange(0, len(conc_data), 2500))
	p = Process(target=create_spectograms_and_store_them, args=(conc_data, i, title, lower, upper, indizes, ))
	processes.append(p)

for thread_id, descriptor in enumerate(data_to_be_used_as_is):
	file_path, lower, upper = descriptor
	conc_data = load(file_path % (i + 1))
	title = file_path.split('/')[-1].split('.')[0]

	print('Length of %s is: [%d]' % (title, len(conc_data)))
	indizes = get_indices_for_noise(np.arange(0, len(conc_data), 2500))
	p = Process(target=create_spectograms_and_store_them, args=(conc_data, i, title, lower, upper, indizes, ))
	processes.append(p)

for p in processes:
	p.start()

for p in processes:
	p.join()

print('Finished by ' + str(datetime.datetime.now()))