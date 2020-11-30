from os import listdir

import matplotlib.pyplot as plt

import numpy as np
from numpy import load

#folder = '/media/christoph/Volume/Masterthesis/histories_of_models_trained_and_saved_test_place/'
for modality in ['csp', 'ccar', 'csp_hil', 'ccar_hil']:
	for folder in ['/media/christoph/Volume/Masterthesis/histories_new_model_trained_and_saved/%s/epoched_intrplt_filt_500_900_kx/' % modality,
		'/media/christoph/Volume/Masterthesis/histories_new_model_trained_and_saved/%s/epoched_intrplt_filt_over_100_kx/' % modality,
		'/media/christoph/Volume/Masterthesis/histories_new_model_trained_and_saved/%s/epoched_intrplt_filt_over_400_kx/' % modality,
		'/media/christoph/Volume/Masterthesis/histories_new_model_trained_and_saved/%s/epoched_intrplt_filt_under_100_kx/' % modality,
		'/media/christoph/Volume/Masterthesis/histories_new_model_trained_and_saved/%s/epoched_intrplt_kx_data_combined/' % modality]:
		for filename in listdir(folder):
			print(filename.replace('.', '-'))
		#-#-#	info = filename.split('_')
		#-#-#	filter_multiplier, learning_rate, filter_size, activation_mod = info[-5:-1]
		#-#-#	filter_multiplier = int(filter_multiplier)
		#-#-#	learning_rate = float(learning_rate)
		#-#-#	filter_size = int(filter_size)
		#-#-#	batch_size = int(info[-1].split('.')[0])
		#-#-#	print(filename + ' w. Conv2D%s, Dense%s, lr=%f' % ('(%d, %d, %d)' % (16*filter_multiplier, 16*filter_multiplier, 32*filter_multiplier), '(%d, %d)' % (256*filter_multiplier, 128*filter_multiplier), learning_rate))
			history = np.load(folder + filename, allow_pickle=True).item()
			fig = plt.figure(figsize=(12, 8))  
			splt_test = plt.subplot(1, 2, 1)  
			splt_val = plt.subplot(1, 2, 2) 
			for key in history:  
				print(key) 
				if 'true' not in key and 'false' not in key:
					if 'val' not in key: 
						splt_test.plot(history[key], label=key, linewidth=3) 
					else: 
						splt_val.plot(history[key], label=key, linewidth=3) 
					for plott in [(splt_test, 'test'), (splt_val, 'val')]: 
						plot, title = plott 
						plot.set_title(title) 
						plot.set_ylabel('Percentage') 
						plot.set_xlabel('Epoch') 
						plot.legend(loc=2) 
						plot.set_ylim([0, 1])
		#	fig.suptitle(filename + '\nw. Conv2D%s of filters (%d, %d), Dense%s, lr=%f and trained w. batch_size of %d items' % ('(%d, %d, %d)' % (16*filter_multiplier, 16*filter_multiplier, 32*filter_multiplier), filter_size, filter_size, '(%d, %d)' % (256*filter_multiplier, 128*filter_multiplier), learning_rate, batch_size))
			fig_title = folder.split('/')[-2] + filename.replace('.', '-')
			fig.suptitle(fig_title)
			plt.savefig('/media/christoph/Volume/Masterthesis/histories_new_model_trained_and_saved/modality_plots/%s/%s' % (modality, fig_title), dpi=80)
			plt.clf()
			plt.close('all')