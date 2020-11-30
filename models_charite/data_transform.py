import numpy as np
from numpy import load

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

hfsep_images_k003 = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_hfsep.npy', allow_pickle=True)
noise_images_k003 = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_noise.npy', allow_pickle=True)
rand_stat = 42
divider_to_save_RAM = 1000

for i in range((hfsep_images_k003.shape[0] // divider_to_save_RAM)):
	X_train, X_test, y_train, y_test = train_test_split(np.concatenate((np.asarray(hfsep_images_k003[i * divider_to_save_RAM:(i + 1)* divider_to_save_RAM]), np.asarray(noise_images_k003[i * divider_to_save_RAM:(i + 1)* divider_to_save_RAM])), axis=0), np.concatenate((np.ones(divider_to_save_RAM, dtype='int8'), np.zeros(divider_to_save_RAM, dtype='int8')), axis=0), test_size=0.33, random_state=rand_stat)

	np.save('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_hfsep_test_part_%d' % i, X_test)
	np.save('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_hfsep_train_part_%d' % i, X_train)
	np.save('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_hfsep_test_part_%d_labels' % i, y_test)
	np.save('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_hfsep_train_part_%d_labels' % i, y_train)

X_train, X_test, y_train, y_test = train_test_split(np.concatenate((np.asarray(hfsep_images_k003[(i + 1) * divider_to_save_RAM:]), np.asarray(noise_images_k003[(i + 1) * divider_to_save_RAM:])), axis=0), np.concatenate((np.ones(hfsep_images_k003.shape[0] - (i * divider_to_save_RAM), dtype='int8'), np.zeros(hfsep_images_k003.shape[0] - (i * divider_to_save_RAM), dtype='int8')), axis=0), test_size=0.33, random_state=rand_stat)
np.save('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_hfsep_test_part_%d' % (i + 1), X_test)
np.save('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_hfsep_train_part_%d' % (i + 1), X_train)
np.save('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_hfsep_test_part_%d_labels' % (i + 1), y_test)
np.save('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_hfsep_train_part_%d_labels' % (i + 1), y_train)







####

from os import listdir
import numpy as np
from numpy import load

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

X_Train = list()
X_Test = list()
y_train = list()
y_test = list()

folder = '/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/'
for filename in listdir(folder):
	if 'ccar_filt_intrplt_filt_500_900_kx_k003_hfsep_test_part_' in filename:  
		if 'labels' in filename:
			y_test.append(np.load(folder + filename, allow_pickle=True))
		else:
			X_Test.append(np.load(folder + filename, allow_pickle=True))
	elif 'ccar_filt_intrplt_filt_500_900_kx_k003_hfsep_train_part_' in filename:
		if 'labels' in filename:
			y_train.append(np.load(folder + filename, allow_pickle=True))
		else:
			X_Train.append(np.load(folder + filename, allow_pickle=True))

X_Train = np.concatenate(X_Train, axis=0)
X_Test = np.concatenate(X_Test, axis=0)
y_train = np.concatenate(y_train, axis=0)
y_test = np.concatenate(y_test, axis=0)

np.save('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_test', X_Test)
np.save('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_train', X_Train)
np.save('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_test_labels', y_test)
np.save('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_train_labels', y_train)



#####

from os import listdir
import numpy as np
from numpy import load

X_Test = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_test.npy', allow_pickle=True)
X_Train = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_train.npy', allow_pickle=True)
y_test = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_test_labels.npy', allow_pickle=True)
y_train = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_train_labels.npy', allow_pickle=True)

