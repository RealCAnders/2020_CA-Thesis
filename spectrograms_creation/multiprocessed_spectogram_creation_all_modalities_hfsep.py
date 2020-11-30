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

def create_spectograms_and_store_them(one_sec_samples_of_ccar_filt_k003, subject_id, title, lower, upper):
	for idx in range(one_sec_samples_of_ccar_filt_k003.shape[1]): 
#		idx = 10
		coors, s = meet.tf.gft(one_sec_samples_of_ccar_filt_k003[:,idx], axis=0, sampling=custom_sampling_meg) 
		t, f, tf_mean_p_interp = meet.tf.interpolate_gft(coors, s, IM_shape=(len(one_sec_samples_of_ccar_filt_k003[:,idx]) // 2, len(one_sec_samples_of_ccar_filt_k003[:,idx])), data_len=len(one_sec_samples_of_ccar_filt_k003[:,idx]), kindf='nearest', kindt='nearest')  
		mean_power_frequency = tf_mean_p_interp.mean(-1)
		for i in range(4): 
#			plt.pcolormesh(t[(2500*i):(2500*(i + 1)) + 1], f[:1001], np.abs(np.multiply(tf_mean_p_interp[:1000, (2500*i):(2500*(i + 1))], np.expand_dims(mean_power_frequency[:1000], 1)))) 
			plt.pcolormesh(t[(2500*i):(2500*(i + 1)) + 1], f[:upper + 1], np.abs(tf_mean_p_interp[:upper, (2500*i):(2500*(i + 1))])) 
			plt.ylim((lower, upper)) 
			plt.colorbar() 
			#plt.savefig('/media/christoph/Volume/Masterthesis/testme/k00%d_%s_%d_%d' % ((subject_id + 1), title, idx, i), dpi=80)
			plt.savefig('/media/christoph/Volume/Masterthesis/spectograms/k00%d/%s_%d_%d' % ((subject_id + 1), title, idx, i), dpi=50) 
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

for i in [1, 2, 3, 4, 0, 5, 6, 7, 8, 9]:

	print('Now working on subject k00%d' % (i + 1))

	processes = []

	for thread_id, descriptor in enumerate(data_to_be_used_as_is):
		file_path, lower, upper = descriptor
		conc_data = load(file_path % (i + 1))
		title = file_path.split('/')[-1].split('.')[0]

		one_sec_win = [0, 10000]
		ccar_filt_k003 = conc_data
		one_sec_samples_of_ccar_filt_k003 = meet.epochEEG(ccar_filt_k003, np.arange(0, len(ccar_filt_k003), 10000), one_sec_win)
		print('Amount of samples for %s is: [%d]' % (title, one_sec_samples_of_ccar_filt_k003.shape[1]))
		p = Process(target=create_spectograms_and_store_them, args=(one_sec_samples_of_ccar_filt_k003, i, title, lower, upper, ))
		processes.append(p)

	for p in processes:
		p.start()

	for p in processes:
		p.join()

print('Finished by ' + str(datetime.datetime.now()))