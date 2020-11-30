from multiprocessing import Process
import meet
import numpy as np
import scipy
import datetime
import matplotlib.pyplot as plt
from numpy import load

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

print('Starting by ' + str(datetime.datetime.now()))

for i in [8, 9]:

	print('Now working on subject k00%d' % (i + 1))

	processes = []

	for thread_id, descriptor in enumerate(data_to_be_analyzed_in_cp5_min_fz):
		file_path, lower, upper = descriptor
		conc_data = load(file_path % (i + 1))
		title = file_path.split('/')[-1].split('.')[0]

		one_sec_win = [0, 10000]
		ccar_filt_k003 = conc_data[5] - conc_data[0]
		one_sec_samples_of_ccar_filt_k003 = meet.epochEEG(ccar_filt_k003, np.arange(0, len(ccar_filt_k003), 10000), one_sec_win)
		print('Amount of samples for %s is: [%d]' % (title, one_sec_samples_of_ccar_filt_k003.shape[1]))
		p = Process(target=create_spectograms_and_store_them, args=(one_sec_samples_of_ccar_filt_k003, i, title, lower, upper, ))
		processes.append(p)

	for p in processes:
		p.start()

	for p in processes:
		p.join()

print('Finished by ' + str(datetime.datetime.now()))

##-#
##-#for file_path in data_to_be_used_as_is:
##-#
##-#	conc_data = load(file_path % (i + 1))
##-#	title = file_path.split('/')[-1].split('.')[0]
##-#
##-#
##-#	mean_sec = one_sec_samples_of_ccar_filt_k003.mean(-1)
##-#	coors, s = meet.tf.gft(mean_sec, axis=0, sampling=custom_sampling_meg)
##-#
##-#	# ToDo: Use sec sample below and go over it four times!
##-#	for idx, sec in enumerate(one_sec_samples_of_ccar_filt_k003.T):
##-#		f, t, tf_mean_p_interp = meet.tf.interpolate_gft(coors, s, IM_shape=(len(sec) // 2, len(sec)), data_len=len(sec), kindf='nearest', kindt='nearest') 
##-#		plt.pcolormesh(f[:2501], t[:1001], np.abs(tf_mean_p_interp[:1000, :2500]))
##-#		plt.colorbar()
##-#		plt.savefig('/media/christoph/Volume/Masterthesis/spectograms/k00%d/%s_%d_0' % ((i + 1), title, idx), dpi=50)
##-#		plt.clf()
##-#		plt.close('all')
##-#		plt.pcolormesh(f[2500:5001], t[:1001], np.abs(tf_mean_p_interp[:1000, 2500:5000]))
##-#		plt.colorbar()
##-#		plt.savefig('/media/christoph/Volume/Masterthesis/spectograms/k00%d/%s_%d_1' % ((i + 1), title, idx), dpi=50)
##-#		plt.clf()
##-#		plt.close('all')
##-#		plt.pcolormesh(f[5000:7501], t[:1001], np.abs(tf_mean_p_interp[:1000, 5000:7500]))
##-#		plt.colorbar()
##-#		plt.savefig('/media/christoph/Volume/Masterthesis/spectograms/k00%d/%s_%d_2' % ((i + 1), title, idx), dpi=50)
##-#		plt.clf()
##-#		plt.close('all')
##-#		plt.pcolormesh(f[7500:], t[:1001], np.abs(tf_mean_p_interp[:1000, 7500:]))
##-#		plt.colorbar()
##-#		plt.savefig('/media/christoph/Volume/Masterthesis/spectograms/k00%d/%s_%d_3' % ((i + 1), title, idx), dpi=50)
##-#		plt.clf()
##-#		plt.close('all')


data_to_be_used_as_is = [

	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_kx_data_combined.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_intrplt_kx_data_combined.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_intrplt_filt_under_100_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_intrplt_filt_over_100_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_intrplt_filt_over_400_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/csp_filt_intrplt_filt_500_900_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_under_100_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_100_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_over_400_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ssd_filt_intrplt_filt_500_900_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_kx_data_combined.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_intrplt_kx_data_combined.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_intrplt_filt_under_100_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_intrplt_filt_over_100_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_intrplt_filt_over_400_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/ccar_filt_intrplt_filt_500_900_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_intrplt_kx_data_combined.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_intrplt_filt_under_100_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_intrplt_filt_over_100_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_intrplt_filt_over_400_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped/k00%d/bcstp_spat_temp_filt_intrplt_filt_500_900_kx.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_intrplt_kx_data_combined_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_intrplt_filt_under_100_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_intrplt_filt_over_100_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_intrplt_filt_over_400_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/csp_filt_intrplt_filt_500_900_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_under_100_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_over_100_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_over_400_kx_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ssd_filt_intrplt_filt_500_900_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_intrplt_kx_data_combined_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_intrplt_filt_under_100_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_intrplt_filt_over_100_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_intrplt_filt_over_400_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/ccar_filt_intrplt_filt_500_900_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_intrplt_filt_under_100_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_intrplt_filt_over_100_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_intrplt_filt_over_400_kx_out_rej.npy',
	'/media/christoph/Volume/Masterthesis/final_prepped_data/advanced_prepped_outliers_rejected/k00%d/bcstp_spat_temp_filt_intrplt_filt_500_900_kx_out_rej.npy'
]
