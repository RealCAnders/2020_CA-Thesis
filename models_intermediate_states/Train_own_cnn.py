#!/usr/bin/python

import sys
import scipy
import pathlib
import shutil
import tempfile
import json
import datetime
import numpy as np
from numpy import load
from numpy import save

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from scipy.fft import fftshift  
from scipy.ndimage import convolve1d, convolve 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import meet

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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

### ### ### ###
### Data gathering
### ### ### ###
base_modalities_per_subject = [
    ['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_kx_data_combined_noise.npy', 'k00%d/epoched_kx_data_combined'],
    ['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_kx_data_combined_noise.npy', 'k00%d/epoched_intrplt_kx_data_combined'],
    ['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_under_100_kx_noise.npy', 'k00%d/epoched_intrplt_filt_under_100_kx'],
    ['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_100_kx_noise.npy', 'k00%d/epoched_intrplt_filt_over_100_kx'],
    ['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_over_400_kx_noise.npy', 'k00%d/epoched_intrplt_filt_over_400_kx'],
    ['/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy', 'k00%d/epoched_intrplt_filt_500_900_kx'],
    ['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_kx_data_combined_out_rej'],
    ['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_under_100_kx_out_rej'],
    ['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_over_100_kx_out_rej'],
    ['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_over_400_kx_out_rej'],
    ['/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_hfsep_out_rej.npy', '/media/christoph/Volume/Masterthesis/final_prepped_data/outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_noise_out_rej.npy', 'outliers_rejected/k00%d/epoched_intrplt_filt_500_900_kx_out_rej'],
]

def calculate_hil_features(transformed):
    hil_dat = scipy.signal.hilbert(transformed, axis=0)
    real_hil_dat = np.real(hil_dat)
    imag_hil_dat = np.imag(hil_dat)
    abs_hil_dat = np.abs(hil_dat)
    angle_hil_dat = np.angle(hil_dat)
    return np.concatenate((real_hil_dat, imag_hil_dat, abs_hil_dat, angle_hil_dat), axis=0)

def load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_path, noise_path, title, identifier):
    # load data in format of (channel x epoch_length x number of epochs)
    title_of_run = title % (identifier + 1)
    hfSEP = load(hfsep_path % (identifier + 1))
    noise = load(noise_path % (identifier + 1))

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
        [np.concatenate((hfSEP[5]-hfSEP[0], noise[5]-noise[0]), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), raw_title],
#       [np.concatenate((hfSEP.reshape(hfSEP.shape[0] * hfSEP.shape[1], -1), noise[:,len(hfSEP)].reshape(noise.shape[0] * noise.shape[1], -1)), axis=1), np.concatenate((hfsep_labels, noise_labels[:,len(hfsep_labels)]), axis=0), title_of_run + '_all_channels_flattened'],
        [np.concatenate((hfSEP_CSP_0, noise_CSP_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), csp_title],
        [np.concatenate((ccar_filt_hfSEP_0, ccar_filt_noise_0), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), ccar_title],
        [np.concatenate((hil_extracted_CSP_hfSEP, hil_extracted_CSP_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_csp_title],
        [np.concatenate((hil_extracted_ccar_hfSEP, hil_extracted_ccar_noise), axis=1), np.concatenate((hfsep_labels, noise_labels), axis=0), hil_ccar_title]
    ]

def create_simple_model(length_of_input):
    input_layer = keras.Input(shape=(length_of_input), name='one_d_input')
    cnn_0 = layers.Conv1D(50, 40, activation='relu')(tf.expand_dims(input_layer, -1))
    den_0 = layers.Dense(35)(cnn_0)
    gap = layers.GlobalAveragePooling1D()(cnn_1)
    drop = layers.Dropout(0.2)(gap)
    dense_out = layers.Dense(1)(drop)

    own_model = keras.Model(inputs=input_layer, outputs=dense_out)

    own_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', 
        metrics=[tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'),
        'accuracy',
        tf.metrics.BinaryAccuracy(threshold=0.0)])

    return own_model

def create_model(length_of_input):
    input_layer = keras.Input(shape=(length_of_input), name='one_d_input')
    cnn_0 = layers.Conv1D(350, 4, activation='relu')(tf.expand_dims(input_layer, -1))
    den_0 = layers.Dense(325)(cnn_0)
    cnn_1 = layers.Conv1D(300, 4, activation='relu')(den_0)
    gap = layers.GlobalAveragePooling1D()(cnn_1)
    drop = layers.Dropout(0.2)(gap)
    dense_out = layers.Dense(1)(drop)

    own_model = keras.Model(inputs=input_layer, outputs=dense_out)

    own_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', 
        metrics=[tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'),
        'accuracy',
        tf.metrics.BinaryAccuracy(threshold=0.0)])

    return own_model

def create_model_longer(length_of_input):
    input_layer = keras.Input(shape=(length_of_input), name='one_d_input')
    cnn_0 = layers.Conv1D(350, 4, activation='relu')(tf.expand_dims(input_layer, -1))
    den_0 = layers.Dense(325)(cnn_0)
    cnn_1 = layers.Conv1D(300, 4, activation='relu')(den_0)
    max_pool = layers.MaxPool1D(3)
    cnn_2 = layers.Conv1D(90, 4, activation='relu')(tf.expand_dims(input_layer, -1))
    den_1 = layers.Dense(70)(cnn_0)
    cnn_3 = layers.Conv1D(50, 4, activation='relu')(den_0)
    gap = layers.GlobalAveragePooling1D()(cnn_1)
    drop = layers.Dropout(0.2)(gap)
    dense_out = layers.Dense(1)(drop)

    own_model = keras.Model(inputs=input_layer, outputs=dense_out)

    own_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', 
        metrics=[tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'),
        'accuracy',
        tf.metrics.BinaryAccuracy(threshold=0.0)])

    return own_model

beginning = datetime.datetime.now()
print('Starting at: %s' % str(beginning))

# conduct subject-dependent training
for i in range(10):

    subject_start = datetime.datetime.now()
    idx = i

    for hfsep_dat, noise_dat, title in base_modalities_per_subject:

        ### 04_ToDo: Classify all the 176 modalities using ELM
        ### Load data of k003, to check on performance and compare w. results pusblished
        workload = load_data_as_still_sorted_hfsep_noise_data_then_labels_from_one_subject(hfsep_dat, noise_dat, title, idx)
        
        for data, labels, run_title in workload:
            print('run_title: %s' % run_title)
            print(data.shape)

            ### Shuffle and split data // .T is required to switch back to shape of (trial x feature)
            shuffled_data, shuffled_labels = shuffle(data.T, labels, random_state=rand_stat)
            print(shuffled_data.shape)
            X_train, X_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.33, random_state=rand_stat)
            print(X_train.shape)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=rand_stat)
            print(X_test.shape)
            print(X_val.shape)

            own_model = create_model(X_train.shape[1])

            ## Train and save the model
            max_epochs = 300
            earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_binary_crossentropy",  
                        mode ="min", patience = 2,  
                        restore_best_weights = True)

            history = own_model.fit(
                x=X_train, 
                y=y_train, 
                epochs=max_epochs, 
                validation_data=(X_val, y_val),
                callbacks =[earlystopping])

            own_model.save('/media/christoph/Volume/Masterthesis/models_trained_and_saved/own_model_1d_%s' % run_title.replace('/', '-'))
            save('/media/christoph/Volume/Masterthesis/histories_of_models_trained_and_saved/own_model_1d_%s' % run_title.replace('/', '-'), history.history)

            long_model = create_model_longer(X_train.shape[1])

            history_long = long_model.fit(
                x=X_train, 
                y=y_train, 
                epochs=max_epochs, 
                validation_data=(X_val, y_val),
                callbacks =[earlystopping])

            long_model.save('/media/christoph/Volume/Masterthesis/models_trained_and_saved/long_model_1d_%s' % run_title.replace('/', '-'))
            save('/media/christoph/Volume/Masterthesis/histories_of_models_trained_and_saved/long_model_1d_%s' % run_title.replace('/', '-'), history_long.history)

            simple_model = create_simple_model(X_train.shape[1])

            history_simple = simple_model.fit(
                x=X_train, 
                y=y_train, 
                epochs=max_epochs, 
                validation_data=(X_val, y_val),
                callbacks =[earlystopping])

            simple_model.save('/media/christoph/Volume/Masterthesis/models_trained_and_saved/simple_model_1d_%s' % run_title.replace('/', '-'))
            save('/media/christoph/Volume/Masterthesis/histories_of_models_trained_and_saved/simple_model_1d_%s' % run_title.replace('/', '-'), history_simple.history)

            # Get the dictionary containing each metric and the loss for each epoch
#            history_dict = history.history
            # Save it under the form of a json file
#            json.dump(history_dict, open('/media/christoph/Volume/Masterthesis/histories_of_models_trained_and_saved/own_model_1d_%s' % run_title.replace('/', '-'), 'w'))
        
    subject_end = datetime.datetime.now()
    print('For subject %d we took %s time' % (idx + 1, str(subject_start - subject_end)))

end = datetime.datetime.now()

print('Overall training time:')
print(beginning - end)

"""
N_VALIDATION = len(x_val)
N_TRAIN = len(x_train)
print('N_TRAIN: %d' % N_TRAIN)
BATCH_SIZE = 491
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

input_layer = keras.Input(shape=(8, 200, 1), name='twodinput')
x_1 = layers.Conv2D(10, (1, 6), activation='relu', name='cond_2d')(input_layer)
x_1_0 = layers.Dense(200, activation='elu')(x_1)
x_2 = layers.MaxPooling2D(pool_size=(4, 5), strides=(1, 1), padding='same') (x_1_0)
x_3 = layers.BatchNormalization(axis = 1)(x_2)
x_4 = layers.Conv2D(5, (4, 50), activation='relu')(x_3)
x_4_0 = layers.Dense(18, activation='relu')(x_4)
x_5 = layers.BatchNormalization(axis = 1)(x_4_0)
x_6 = layers.Conv2D(5, (3, 70), activation='relu')(x_5)
x_7 = layers.MaxPooling2D(pool_size=(2, 40), strides=(1, 1), padding='same') (x_6)
x_8 = layers.Flatten()(x_7)
x_9 = layers.Dense(10, name = 'pre_out')(x_8)
x_10 = layers.Dense(1, name = 'out')(x_9)
two_d_outputs = layers.Activation('sigmoid', name='sigmoid')(x_10)

own_model = keras.Model(inputs=input_layer, outputs=two_d_outputs)

METRICS = [
    tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'),
	'accuracy',
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

# visualize learning rate: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit?hl=en
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
	0.001,
	decay_steps=STEPS_PER_EPOCH*1000,
	decay_rate=1,
	staircase=False)

def get_optimizer():
	return tf.keras.optimizers.Adam(lr_schedule)

class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_test_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )

own_model.compile(optimizer=get_optimizer(), 
               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
               metrics=METRICS)

own_model.summary()

## Train and save the model
max_iterations_for_training = 10000
epochs_per_loop_iteration = 20

history = own_model.fit(
	x=x_train, 
	y=y_train, 
	steps_per_epoch=STEPS_PER_EPOCH,
	epochs=epochs_per_loop_iteration, 
	validation_data=(x_val, y_val),
	callbacks=get_callbacks('experimental_parameter_derivation/own_cnn'),
	verbose=0)

own_model.save('/home/christoph/Desktop/Beginning_September_Work/scripts/own_model_baseline_prepped_data')


plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.savefig('/home/christoph/Desktop/Beginning_September_Work/scripts/own_model_baseline_prepped_data_performance')
"""