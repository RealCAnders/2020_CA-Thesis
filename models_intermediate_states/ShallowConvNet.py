from __future__ import print_function 
from numpy import load                                                                                                                                              
import numpy as np                                                                                                                                                  
from os import listdir 
import numpy as np 
from numpy import load 
import sys 

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

#-#-# X_Test = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_test.npy', allow_pickle=True)[:,29:-26,40:-82,:] / 255.0 
#-#-# y_test = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_test_labels.npy', allow_pickle=True) 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, models 
 
import tensorflow_docs as tfdocs 
import tensorflow_docs.modeling 
import tensorflow_docs.plots 
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)

import time 
 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
 
import numpy as np 
np.random.seed(1234) 
from functools import reduce 
import math as m 

import scipy
import scipy.io 
import meet

from pyts.image import RecurrencePlot


rand_stat = 42


def normalize_z(to_be_normalized): 
    return (to_be_normalized - np.mean(to_be_normalized)) / (np.std(to_be_normalized))


def calculate_hil_features(transformed):
	hil_dat = scipy.signal.hilbert(transformed, axis=0)
	real_hil_dat = np.real(hil_dat)
	imag_hil_dat = np.imag(hil_dat)
	abs_hil_dat = np.abs(hil_dat)
	angle_hil_dat = np.angle(hil_dat)
	return np.asarray((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat))


def metrics_for_conf_mat(tn, fp, fn, tp):
    """
    Computes for given confusion-matrix entries the metrics
    Sensitivity, Specificity, Accuracy, F1-Score and MCC 
    More info: https://en.wikipedia.org/wiki/Confusion_matrix
    """
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = (2 * tp) / ((2 * tp) + fp + fn)
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (sensitivity, specificity, accuracy, f1_score, mcc)


k = int(sys.argv[1])
title = str(sys.argv[2])

rp = RecurrencePlot()
model = keras.models.Sequential([  
    keras.layers.Conv2D(filters=40, kernel_size=(10,1), input_shape=(250,250,1)),  
    keras.layers.Conv2D(filters=40, kernel_size=(40,44), activation='elu'),  
    keras.layers.AveragePooling2D(pool_size=(75,1), strides=(15,1)),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation='sigmoid')  
]) 

METRICS = [
   tf.keras.metrics.TruePositives(),
   tf.keras.metrics.FalsePositives(),
   tf.keras.metrics.TrueNegatives(),
   tf.keras.metrics.FalseNegatives(), 
   tf.keras.metrics.BinaryAccuracy(),
   tf.keras.metrics.Precision(),
   tf.keras.metrics.Recall(),
   tf.keras.metrics.AUC(),
   tf.keras.metrics.MeanAbsoluteError(),
]

model.compile(optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9, clipvalue=5.0),  
   loss=tf.keras.losses.BinaryCrossentropy(), 
   metrics=METRICS)

noise_k00x = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/%s_noise.npy' % (k, title))
hfsep_k00x = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/%s_hfsep.npy' % (k, title))

csp_filters, csp_eigenvals = meet.spatfilt.CSP(hfsep_k00x[:8,:,:].mean(2), noise_k00x[:8,:,:].mean(2))
hfSEP_CSP_0 = np.tensordot(csp_filters[0].T, hfsep_k00x[:8,:,:], axes=(0 ,0))
noise_CSP_0 = np.tensordot(csp_filters[0].T, noise_k00x[:8,:,:], axes=(0 ,0))

znorm_noise = normalize_z(noise_CSP_0)
znorm_hfsep = normalize_z(hfSEP_CSP_0)

print(znorm_hfsep.shape)

X_train, X_test, y_train, y_test = train_test_split(
    np.swapaxes(np.concatenate((znorm_hfsep, znorm_noise), axis=1), 1, 0),
    np.concatenate((np.ones(znorm_hfsep.shape[-1], dtype='int8'), np.zeros(znorm_noise.shape[-1], dtype='int8')), axis=0),
    test_size=0.33,
    random_state=rand_stat)

print(X_train.shape)

X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.33, random_state=rand_stat)

# Remove dimensions to assure dividability:
X_train = X_train[:(32 * (X_train.shape[0] // 32))]
X_test = X_test[:(32 * (X_test.shape[0] // 32))]
X_eval = X_eval[:(32 * (X_eval.shape[0] // 32))]

X_train = np.expand_dims(rp.fit_transform(X_train), axis=3)
X_test = np.expand_dims(rp.fit_transform(X_test), axis=3)
X_eval = np.expand_dims(rp.fit_transform(X_eval), axis=3)

y_train = y_train[:(32 * (y_train.shape[0] // 32))]
y_test = y_test[:(32 * (y_test.shape[0] // 32))]
y_eval = y_eval[:(32 * (y_eval.shape[0] // 32))]

# Our vectorized labels
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))
y_eval = y_eval.reshape((-1,1))

history = model.fit(x=X_train, y=y_train, epochs=25, batch_size=32, validation_data=[X_eval, y_eval]) 

model.save('/media/christoph/Volume/Masterthesis/recurrence_plot_training/within_subject/shallow_conv_net/models_trained_and_saved_test_place/%s/shallow_conv_net_w_%s_lr_%f_clip_%s_k00%d_on_csp' % (title, 'sgd', 0.001, 'True', k))
np.save('/media/christoph/Volume/Masterthesis/recurrence_plot_training/within_subject/shallow_conv_net/histories_of_models_trained_and_saved_test_place/%s/shallow_conv_net_w_%s_lr_%f_clip_%s_k00%d_on_csp' % (title, 'sgd', 0.001, 'True', k), history.history)

predictions = np.abs(np.rint(model.predict(x=X_test)))
confusion_matrix_for_model = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = confusion_matrix_for_model.ravel()
np.save('/media/christoph/Volume/Masterthesis/recurrence_plot_training/within_subject/shallow_conv_net/histories_of_models_trained_and_saved_test_place_confusion_matrices/%s/shallow_conv_net_w_sgd_lr_0001_clip_True_k00%d_tn_%d_fp_%d_fn_%d_tp_%d_on_csp_confusion_matrix' % (title, k, tn, fp, fn, tp), confusion_matrix_for_model)

metrics = metrics_for_conf_mat(tn, fp, fn, tp)
np.save('/media/christoph/Volume/Masterthesis/recurrence_plot_training/within_subject/shallow_conv_net/histories_of_models_trained_and_saved_test_place_confusion_matrices/%s/shallow_conv_net_w_sgd_lr_0001_clip_True_k00%d_tn_%d_fp_%d_fn_%d_tp_%d_on_csp_metrics' % (title, k, tn, fp, fn, tp), metrics)