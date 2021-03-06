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

hfsep_dat = str(sys.argv[1])
noise_dat = str(sys.argv[2])
title = str(sys.argv[3])
idx = int(sys.argv[4])

rp = RecurrencePlot()
model = keras.models.Sequential([  
    keras.layers.Conv2D(filters=25, kernel_size=(10,1), input_shape=(250,250,1)),  
    keras.layers.Conv2D(filters=25, kernel_size=(25,44), activation='elu'),  
    keras.layers.MaxPool2D(pool_size=(3,1), strides=(3,1)), 
    keras.layers.Conv2D(filters=50, kernel_size=(10,25), strides=(1,1), activation='elu', padding="valid"), 
    keras.layers.MaxPool2D(pool_size=(3,1), strides=(3,1)), 
    keras.layers.Conv2D(filters=100, kernel_size=(10,50), strides=(1,1), activation='elu', padding="valid"), 
    keras.layers.MaxPool2D(pool_size=(3,1), strides=(3,1)), 
    keras.layers.Conv2D(filters=200, kernel_size=(2,100), strides=(1,1), activation='elu', padding="valid"), 
    keras.layers.MaxPool2D(pool_size=(3,1), strides=(3,1)), 
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation='sigmoid')  
]) 

# could also try this,... has just same padding as opposed to valid padding...
# model = keras.models.Sequential([ 
#     keras.layers.Conv2D(filters=25, kernel_size=(10,1), input_shape=(250,250,1)), 
#     keras.layers.Conv2D(filters=25, kernel_size=(25,44), activation='elu'), 
#     keras.layers.MaxPool2D(pool_size=(3,1), strides=(3,1)),
#     keras.layers.Conv2D(filters=50, kernel_size=(10,25), strides=(1,1), activation='elu', padding="same"),
#     keras.layers.MaxPool2D(pool_size=(3,1), strides=(3,1)),
#     keras.layers.Conv2D(filters=100, kernel_size=(10,50), strides=(1,1), activation='elu', padding="same"),
#     keras.layers.MaxPool2D(pool_size=(3,1), strides=(3,1)),
#     keras.layers.Conv2D(filters=200, kernel_size=(10,100), strides=(1,1), activation='elu', padding="same"),
#     keras.layers.MaxPool2D(pool_size=(3,1), strides=(3,1)),
#     keras.layers.Dense(1, activation='sigmoid') 
# ])



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

a_ccar, b_ccar, s_ccar = meet.spatfilt.CCAvReg(hfsep_k00x[:8,:,:])
ccar_filt_hfSEP_0 = np.tensordot(a_ccar[:,0], hfsep_k00x[:8,:,:], axes=(0, 0))
ccar_filt_noise_0 = np.tensordot(a_ccar[:,0], hfsep_k00x[:8,:,:], axes=(0, 0))

znorm_noise = normalize_z(ccar_filt_noise_0)
znorm_hfsep = normalize_z(ccar_filt_hfSEP_0)

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

model.save('/media/christoph/Volume/Masterthesis/recurrence_plot_training/within_subject/dense_conv_net/models_trained_and_saved_test_place/%s/dense_conv_net_w_%s_lr_%f_clip_%s_k00%d_on_cccar' % (title, 'sgd', 0.001, 'True', k))
np.save('/media/christoph/Volume/Masterthesis/recurrence_plot_training/within_subject/dense_conv_net/histories_of_models_trained_and_saved_test_place/%s/dense_conv_net_w_%s_lr_%f_clip_%s_k00%d_on_cccar' % (title, 'sgd', 0.001, 'True', k), history.history)

predictions = np.abs(np.rint(model.predict(x=X_test)))
confusion_matrix_for_model = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = confusion_matrix_for_model.ravel()
np.save('/media/christoph/Volume/Masterthesis/recurrence_plot_training/within_subject/dense_conv_net/histories_of_models_trained_and_saved_test_place_confusion_matrices/%s/dense_conv_net_w_sgd_lr_0001_clip_True_k00%d_tn_%d_fp_%d_fn_%d_tp_%d_on_cccar_confusion_matrix' % (title, k, tn, fp, fn, tp), confusion_matrix_for_model)

metrics = metrics_for_conf_mat(tn, fp, fn, tp)
np.save('/media/christoph/Volume/Masterthesis/recurrence_plot_training/within_subject/dense_conv_net/histories_of_models_trained_and_saved_test_place_confusion_matrices/%s/dense_conv_net_w_sgd_lr_0001_clip_True_k00%d_tn_%d_fp_%d_fn_%d_tp_%d_on_cccar_metrics' % (title, k, tn, fp, fn, tp), metrics)



###############
###############
###############

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
import pickle
from scipy.fft import fftshift  
from scipy.ndimage import convolve1d, convolve 

from pyts.image import RecurrencePlot


rand_stat = 42


hfsep_dat = str(sys.argv[1])
noise_dat = str(sys.argv[2])
title = str(sys.argv[3])
idx = int(sys.argv[4])


def normalize_z(to_be_normalized): 
    return (to_be_normalized - np.mean(to_be_normalized)) / (np.std(to_be_normalized))


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


args = sys.argv
print('Number of arguments: %d arguments.' % len(args))
print('Argument List:', str(args))

rand_stat = 42

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


def calculate_hil_features(transformed):
  hil_dat = scipy.signal.hilbert(transformed, axis=0)
  real_hil_dat = np.real(hil_dat)
  imag_hil_dat = np.imag(hil_dat)
  abs_hil_dat = np.abs(hil_dat)
  angle_hil_dat = np.angle(hil_dat)
  return np.stack((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat), axis=0)


def load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_path, noise_path, title, identifier):
  # load data in format of (channel x epoch_length x number of epochs)
  title_of_run = title % (identifier)
  hfSEP = load(hfsep_path % (identifier))
  noise = load(noise_path % (identifier))

  ### 01_ToDo: Modify all the 11 base-modalities through: [SSD Y/N] (load respective modifiers therefore)
  # Still will remain open. Only thing to do here however: Load non-epoched data, compute SSD, epoch data
  raw_title = title_of_run + '_raw'

  ### 02_ToDo: Modify all the 22 modalities through: [leave, CSP, CCAr, bCSTP]
  # Compute CSP
  csp_title = title_of_run + '_CSP'
  csp_filters, csp_eigenvals = meet.spatfilt.CSP(hfSEP[:8,:,:].mean(2), noise[:8,:,:].mean(2))
  hfSEP_CSP_0 = np.tensordot(csp_filters[0].T, hfSEP[:8,:,:], axes=(0 ,0))
  noise_CSP_0 = np.tensordot(csp_filters[0].T, noise[:8,:,:], axes=(0 ,0))
  hfSEP_CSP_1 = np.tensordot(csp_filters[1].T, hfSEP[:8,:,:], axes=(0 ,0))
  noise_CSP_1 = np.tensordot(csp_filters[1].T, noise[:8,:,:], axes=(0 ,0))
  hfSEP_CSP_2 = np.tensordot(csp_filters[2].T, hfSEP[:8,:,:], axes=(0 ,0))
  noise_CSP_2 = np.tensordot(csp_filters[2].T, noise[:8,:,:], axes=(0 ,0))

  # Compute CCAr
  ccar_title = title_of_run + '_CCAr'
  a_ccar, b_ccar, s_ccar = meet.spatfilt.CCAvReg(hfSEP[:8,:,:])
  ccar_filt_hfSEP_0 = np.tensordot(a_ccar[:,0], hfSEP[:8,:,:], axes=(0, 0))
  ccar_filt_noise_0 = np.tensordot(a_ccar[:,0], noise[:8,:,:], axes=(0, 0))
  ccar_filt_hfSEP_1 = np.tensordot(a_ccar[:,1], hfSEP[:8,:,:], axes=(0, 0))
  ccar_filt_noise_1 = np.tensordot(a_ccar[:,1], noise[:8,:,:], axes=(0, 0))
  ccar_filt_hfSEP_2 = np.tensordot(a_ccar[:,2], hfSEP[:8,:,:], axes=(0, 0))
  ccar_filt_noise_2 = np.tensordot(a_ccar[:,2], noise[:8,:,:], axes=(0, 0))

  # Compute bCSTP
  # s_bcstp_eigenvals, t_bcstp_eigenvals, W_bcstp, V_bcstp = bCSTP(hfSEP[:8,:,:], noise[:8,:,:], num_iter=15, t_keep=3, s_keep=3)
  # left out as it would also need intrplt.data.... the scipy.ndimage.convolve1d(np.dot(W_out_epoched_intrplt_kx_data_combined_hfsep[-1][:,0], intrplt_kx_data_combined[:8]), V_out_epoched_intrplt_kx_data_combined_hfsep[-1][:,0][::-1], axis=-1)
  
  ### 03_ToDo: Modify all the 88 modalities through: [hil Y/N]
  hil_csp_title = title_of_run + '_CSP_hil'
  hil_extracted_hfSEP_CSP_0 = calculate_hil_features(hfSEP_CSP_0)
  hil_extracted_noise_CSP_0 = calculate_hil_features(noise_CSP_0)
  hil_extracted_hfSEP_CSP_1 = calculate_hil_features(hfSEP_CSP_1)
  hil_extracted_noise_CSP_1 = calculate_hil_features(noise_CSP_1)
  hil_extracted_hfSEP_CSP_2 = calculate_hil_features(hfSEP_CSP_2)
  hil_extracted_noise_CSP_2 = calculate_hil_features(noise_CSP_2)
  hil_extracted_CSP_hfSEP = np.concatenate((hil_extracted_hfSEP_CSP_0, hil_extracted_hfSEP_CSP_1, hil_extracted_hfSEP_CSP_2), axis=0)
  hil_extracted_CSP_noise = np.concatenate((hil_extracted_noise_CSP_0, hil_extracted_noise_CSP_1, hil_extracted_noise_CSP_2), axis=0)

  hil_ccar_title = title_of_run + '_CCAR_hil'
  hil_extracted_ccar_filt_hfSEP_0 = calculate_hil_features(ccar_filt_hfSEP_0)
  hil_extracted_ccar_filt_noise_0 = calculate_hil_features(ccar_filt_noise_0)
  hil_extracted_ccar_filt_hfSEP_1 = calculate_hil_features(ccar_filt_hfSEP_1)
  hil_extracted_ccar_filt_noise_1 = calculate_hil_features(ccar_filt_noise_1)
  hil_extracted_ccar_filt_hfSEP_2 = calculate_hil_features(ccar_filt_hfSEP_2)
  hil_extracted_ccar_filt_noise_2 = calculate_hil_features(ccar_filt_noise_2)
  hil_extracted_ccar_hfSEP = np.concatenate((hil_extracted_ccar_filt_hfSEP_0, hil_extracted_ccar_filt_hfSEP_1, hil_extracted_ccar_filt_hfSEP_2), axis=0)
  hil_extracted_ccar_noise = np.concatenate((hil_extracted_ccar_filt_noise_0, hil_extracted_ccar_filt_noise_1, hil_extracted_ccar_filt_noise_2), axis=0)

  hfsep_labels = np.ones(len(hfSEP[0,0,:]), dtype=np.int8)
  noise_labels = np.zeros(len(noise[0,0,:]), dtype=np.int8)

  # return the datasets in epoch, channel, time_in_channel - fashion
  return [
    [np.concatenate((hfSEP[:8], noise[:8]), axis=-1), np.concatenate((hfsep_labels, noise_labels), axis=0), raw_title],
#   [np.concatenate((hfSEP.reshape(-1, hfSEP.shape[-1]), noise.reshape(-1, noise.shape[-1])), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), title_of_run + '_all_channels_flattened'],// I have no idea why, but this singular error still occurs
    [np.concatenate((np.stack((hfSEP_CSP_0, hfSEP_CSP_1, hfSEP_CSP_2), axis=0), np.stack((noise_CSP_0, noise_CSP_1, noise_CSP_2), axis=0)), axis=2), np.concatenate((hfsep_labels, noise_labels), axis=0), csp_title],
    [np.concatenate((np.stack((ccar_filt_hfSEP_0, ccar_filt_hfSEP_1, ccar_filt_hfSEP_2), axis=0), np.stack((ccar_filt_noise_0, ccar_filt_noise_1, ccar_filt_noise_2), axis=0)), axis=2), np.concatenate((hfsep_labels, noise_labels), axis=0), ccar_title],
    [np.concatenate((hil_extracted_CSP_hfSEP, hil_extracted_CSP_noise), axis=2), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_csp_title],
    [np.concatenate((hil_extracted_ccar_hfSEP, hil_extracted_ccar_noise), axis=2), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_ccar_title]
  ]


def create_mc_cnn_for_2d(input_len, amount_of_channels):

  # Define the respective channels for the electrodes signals
  channel_models = []
  input_channels = []
  for channel_id in range(amount_of_channels):
    input_channel = tf.keras.Input((input_len, 1), name=('Channel-%d' % channel_id))
    c_conv = tf.keras.layers.Conv1D(32, 3, use_bias=False)(input_channel)
    c_pool = tf.keras.layers.AveragePooling1D(pool_size=2)(c_conv)
    c_conv = tf.keras.layers.Conv1D(16, 4, use_bias=False)(c_pool)
    c_pool = tf.keras.layers.MaxPool1D(pool_size=2)(c_conv)
    out = tf.keras.layers.Activation('sigmoid')(c_pool)
    channel_model = tf.keras.Model(inputs=input_channel, outputs=out)
  
    input_channels.append(input_channel)    
    channel_models.append(channel_model)
  
  # concatente the 1D-channels to a MC-Model
  combined_model = tf.keras.layers.concatenate(
      [tf.keras.backend.expand_dims(channl.output, axis=1) for channl in channel_models]
      , axis=1
  )
  
  # further define the MC-Model using Conv2D, Dense, Pooling and Flattening
  conv = tf.keras.layers.Conv2D(128, (2, 5))(combined_model)
  den = tf.keras.layers.Dense(512)(conv)
  conv = tf.keras.layers.Conv2D(128, (2, 15))(den)
  den = tf.keras.layers.Dense(512)(conv)
  conv = tf.keras.layers.Conv2D(64, (1, 10))(den)
  avg = tf.keras.layers.AveragePooling2D((1, 8))(conv)
  flat = tf.keras.layers.Flatten()(avg)
  drop = tf.keras.layers.Dropout(0.5)(flat)
  den = tf.keras.layers.Dense(10)(drop)
  out = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(den)
  
  # Build the final model, show a summary and compile it w. metrics
  mc_cnn = tf.keras.Model(inputs=input_channels, outputs=out)

  mc_cnn.summary()

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

  mc_cnn.compile(optimizer=tf.keras.optimizers.SGD(0.001, momentum=0.9, clipvalue=5.0),  
    loss=tf.keras.losses.BinaryCrossentropy(), 
    metrics=METRICS)

  return mc_cnn


workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_dat, noise_dat, title, idx)

for data, labels, run_title in workload:
  print('run_title: %s' % run_title)
  print(data.shape)

  ### Shuffle and split data // .T is required to switch back to shape of (trial x feature)
  shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat)
  print(shuffled_data.shape)
  X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat)
  print(X_train.shape)
  X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.33, random_state=rand_stat)

  X_train = np.expand_dims(np.swapaxes(X_train, 1, 2), axis=-1)
  X_test = np.expand_dims(np.swapaxes(X_test, 1, 2), axis=-1)
  X_eval = np.expand_dims(np.swapaxes(X_eval, 1, 2), axis=-1)

  y_train = y_train.reshape((-1,1))
  y_test = y_test.reshape((-1,1))
  y_eval = y_eval.reshape((-1,1))

  # some model definition and training
  model = create_mc_cnn_for_2d(X_train.shape[2], X_train.shape[1])

  history = model.fit(x=[np.squeeze(x, axis=1) for x in np.split(X_train, indices_or_sections=X_train.shape[1], axis=1)], y=y_train, epochs=25, batch_size=32, validation_data=[[np.squeeze(x, axis=1) for x in np.split(X_eval, indices_or_sections=X_eval.shape[1], axis=1)], y_eval]) 

  model.save('/media/christoph/Volume/Masterthesis/2d_tsc_november/mc_cnn/models/%d/mc_cnn_on_%s' % (idx, run_title.replace('/', '-')))
  np.save('/media/christoph/Volume/Masterthesis/2d_tsc_november/mc_cnn/history/%d/mc_cnn_on_%s' % (idx, run_title.replace('/', '-')), history.history)

  predictions = np.abs(np.rint(model.predict(x=[np.squeeze(x, axis=1) for x in np.split(X_test, indices_or_sections=X_test.shape[1], axis=1)])))
  confusion_matrix_for_model = confusion_matrix(y_test, predictions)
  tn, fp, fn, tp = confusion_matrix_for_model.ravel()
  np.save('/media/christoph/Volume/Masterthesis/2d_tsc_november/mc_cnn/%d/mc_cnn_on_%s_confusion_matrix' % (idx, run_title.replace('/', '-')), confusion_matrix_for_model)

  metrics = metrics_for_conf_mat(tn, fp, fn, tp)
  np.save('/media/christoph/Volume/Masterthesis/2d_tsc_november/mc_cnn/%d/mc_cnn_on_%s_metrics' % (idx, run_title.replace('/', '-')), metrics)
