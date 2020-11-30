from os import listdir

import matplotlib.pyplot as plt

import numpy as np
from numpy import load

folder = '/media/christoph/Volume/Masterthesis/histories_new_model_trained_and_saved/epoched_intrplt_filt_500_900_kx/'
for filename in listdir(folder):
	print(filename.replace('.', '-'))
	history = np.load(folder + filename, allow_pickle=True).item()
	for key in history:  
		if 'true' in key or 'false' in key:
			if 'val' in key: 
				print(key + ' == ' + str(history[key][-1]))