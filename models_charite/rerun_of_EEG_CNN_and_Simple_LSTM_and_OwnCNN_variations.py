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

rand_stat = 42

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


def build_EEG_CNN(lr, img_shape, input_shape):
    ## Some model parameters and model definition
    # shape of the image (SHAPE x SHAPE)
    shapex, shapey = img_shape
    # number of convolutional filters to use
    nb_filters = 32
    # level of pooling to perform (POOL x POOL)
    nb_pool = 2
    # level of convolution to perform (CONV x CONV)
    nb_conv = 1
    # amount of classes
    nb_classes = 2

    model = models.Sequential()

    model.add(keras.Input(shape = (250, 8, 1)))
    model.add(layers.Conv2D(filters=nb_filters, kernel_size=1, strides=(nb_conv, nb_conv), padding="valid"))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters=nb_filters, kernel_size=2, strides=(nb_conv, nb_conv)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D(pool_size=(nb_pool, nb_pool)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    # the resulting image after conv and pooling is the original shape
    # divided by the pooling with a number of filters for each "pixel"
    # (the number of filters is determined by the last Conv2D)
    model.add(layers.Dense(nb_filters * (shapex / nb_pool) * (shapey / nb_pool)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    model.summary()

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

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, clipvalue=5.0),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=METRICS)

    return model


#-#-#class SpatialSplit(tf.keras.layers.Layer):
#-#-#    """Split the input tensor into 8 tensors along the spatial dimension."""
#-#-#
#-#-#
#-#-#    def call(self, inputs):
#-#-#        # Expect the input to be 3D and mask to be 2D, split the input tensor into 8
#-#-#        # subtensors along the time axis (axis 1).
#-#-#        print("INSIDE OF SPLIT")
#-#-#        print(inputs)
#-#-#        print(inputs.shape)
#-#-#        return tf.split(inputs, 8, axis=2)
#-#-#
#-#-#
#-#-#    def compute_mask(self, inputs, mask=None):
#-#-#        # Also split the mask into 8 if it presents.
#-#-#        if mask is None:
#-#-#            return None
#-#-#        return tf.split(mask, 8, axis=2)

#-#-#   ## Not used but could be a starting point with:         # LSTM: lstm = build_LSTM(0.001, (250, 8), (250, 8))
#-#-#def build_LSTM(lr, img_shape, input_shape):
#-#-#
#-#-#    time_lenght = 250
#-#-#
#-#-#    # Define the respective channels for the electrodes signals
#-#-#    input_layer = tf.keras.Input(shape=input_shape)
#-#-#    #-#split_layer_out = SpatialSplit()(input_layer)
#-#-#    #-#channel_models = []
#-#-#
#-#-#    lstm_one = layers.LSTM(128)(input_layer)
#-#-#    sigmoid_lstm_out = tf.keras.layers.Activation('sigmoid')(lstm_one)
#-#-#    reshaped = layers.Reshape((4, 32))(sigmoid_lstm_out)
#-#-#    conv_one = layers.Conv1D(128, 2)(reshaped)
#-#-#    conv_one_1 = layers.Conv1D(64, 2)(conv_one)
#-#-#    conv_one_2 = layers.Conv1D(32, 1)(conv_one_1)
#-#-#    flatten = layers.Flatten()(conv_one_2)
#-#-#    dense = layers.Dense(10)(flatten)
#-#-#    drop = layers.Dropout(0.5)(dense)
#-#-#    dense_1 = layers.Dense(1)(drop)
#-#-#    sigmoid = layers.Activation('sigmoid', name = 'sigmoid')(dense_1)
#-#-#
#-#-#    model = keras.Model(inputs=input_layer, outputs=sigmoid)
#-#-#
#-#-#    model.summary()
#-#-#
#-#-#    METRICS = [
#-#-#        tf.keras.metrics.TruePositives(),
#-#-#        tf.keras.metrics.FalsePositives(),
#-#-#        tf.keras.metrics.TrueNegatives(),
#-#-#        tf.keras.metrics.FalseNegatives(), 
#-#-#        tf.keras.metrics.BinaryAccuracy(),
#-#-#        tf.keras.metrics.Precision(),
#-#-#        tf.keras.metrics.Recall(),
#-#-#        tf.keras.metrics.AUC(),
#-#-#        tf.keras.metrics.MeanAbsoluteError(),
#-#-#    ]
#-#-#
#-#-#    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, clipvalue=5.0),
#-#-#                  loss=tf.keras.losses.BinaryCrossentropy(),
#-#-#                  metrics=METRICS)
#-#-#
#-#-#    return model





for title in ['epoched_intrplt_kx_data_combined',
    'epoched_intrplt_filt_under_100_kx',
    'epoched_intrplt_filt_over_100_kx',
    'epoched_intrplt_filt_over_400_kx',
    'epoched_intrplt_filt_500_900_kx']:
    for k in range(1, 11): 

        ## Data loading and transformation
        noise_k00x = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/%s_noise.npy' % (k, title))
        hfsep_k00x = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/%s_hfsep.npy' % (k, title))

        znorm_noise = normalize_z(noise_k00x[:8])
        znorm_hfsep = normalize_z(hfsep_k00x[:8])

        X_train, X_test, y_train, y_test = train_test_split(
            np.swapaxes(np.concatenate((znorm_hfsep, znorm_noise), axis=2), 2, 0),
            np.concatenate((np.ones(znorm_hfsep.shape[-1], dtype='int8'), np.zeros(znorm_noise.shape[-1], dtype='int8')), axis=0),
            test_size=0.33,
            random_state=rand_stat)

        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.33, random_state=rand_stat)

        # Remove dimensions to assure dividability:
        X_train = np.expand_dims(X_train[:(32 * (X_train.shape[0] // 32))], axis=-1)
        X_test = np.expand_dims(X_test[:(32 * (X_test.shape[0] // 32))], axis=-1)
        X_eval = np.expand_dims(X_eval[:(32 * (X_eval.shape[0] // 32))], axis=-1)

        y_train = y_train[:(32 * (y_train.shape[0] // 32))]
        y_test = y_test[:(32 * (y_test.shape[0] // 32))]
        y_eval = y_eval[:(32 * (y_eval.shape[0] // 32))]

        # Our vectorized labels
        y_train = y_train.reshape((-1,1))
        y_test = y_test.reshape((-1,1))
        y_eval = y_eval.reshape((-1,1))


        ## Model definitions
        ## For in_shape=(250, 8)
        # EEG_CNN: eeg_cnn = build_EEG_CNN(0.001, (250, 8), (250, 8, 1)) 

        eeg_cnn = build_EEG_CNN(0.001, (250, 8), (250, 8, 1)) 

        history = eeg_cnn.fit(x=X_train, y=y_train, epochs=25, batch_size=32, validation_data=[X_eval, y_eval]) 

        model.save('/media/christoph/Volume/Masterthesis/eeg_cnn/eeg_cnn_k00%d_model' % k)
        np.save('/media/christoph/Volume/Masterthesis/histories_eeg_cnn/eeg_cnn_k00%d_history' % k, history.history)

        predictions = np.abs(np.rint(model.predict(x=X_test)))
        confusion_matrix_for_model = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = confusion_matrix_for_model.ravel()
        np.save('/media/christoph/Volume/Masterthesis/eeg_cnn/eeg_cnn_k00%d_tn_%d_fp_%d_fn_%d_tp_%d_confusion_matrix' % (k, tn, fp, fn, tp), confusion_matrix_for_model)

        metrics = metrics_for_conf_mat(tn, fp, fn, tp)
        np.save('/media/christoph/Volume/Masterthesis/eeg_cnn/eeg_cnn_k00%d_tn_%d_fp_%d_fn_%d_tp_%d_metrics' % (k, tn, fp, fn, tp), metrics)
