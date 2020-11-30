#!/usr/bin/python

import sys
import numpy as np
from numpy import load

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

args = sys.argv

print('Number of arguments: %d arguments.' % len(args))
print('Argument List:', str(args))

## Some model definition
def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
    advised to do some model searching to get optimal performance on your
    particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = keras.Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = layers.Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (1, Chans, Samples),
                                   use_bias = False)(input1)
    block1       = layers.BatchNormalization(axis = 1)(block1)
    block1       = layers.DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = tf.keras.constraints.max_norm(1.))(block1)
    block1       = layers.BatchNormalization(axis = 1)(block1)
    block1       = layers.Activation('elu')(block1)
    block1       = layers.AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = layers.SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = layers.BatchNormalization(axis = 1)(block2)
    block2       = layers.Activation('elu')(block2)
    block2       = layers.AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = layers.Flatten(name = 'flatten')(block2)
    
    dense        = layers.Dense(1, name = 'dense', 
                         kernel_constraint = tf.keras.constraints.max_norm(norm_rate))(flatten)
    sigmoid      = layers.Activation('sigmoid', name = 'sigmoid')(dense)
    
    return keras.Model(inputs=input1, outputs=sigmoid)


## Create the model
eeg_net = EEGNet(2, 8, 200)

eeg_net.summary()

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

eeg_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),  # Optimizer
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS)

data_hfsep = load("/home/christoph/Desktop/Beginning_September_Work/Prepped_Data/OutliersRejected/k001_500_900_int_70_70_hfsep_100_300.npy")
print(data_hfsep.shape)
data_noise = load("/home/christoph/Desktop/Beginning_September_Work/Prepped_Data/OutliersRejected/k001_500_900_int_70_70_noise_100_300.npy")
print(data_noise.shape)

## Combine data that has shape: channels x time_datapoint x sample
x_train_hfsep = data_hfsep[:8,:,:-500]
y_train_hfsep = np.ones(len(x_train_hfsep[0,0,:]))
x_train_noise = data_noise[:8,:,:-500]
y_train_noise = np.zeros(len(x_train_noise[0,0,:]))

x_test_hfsep = data_hfsep[:8,:,-100:]
y_test_hfsep = np.ones(len(x_test_hfsep[0,0,:]))
x_test_noise = data_noise[:8,:,-100:]
y_test_noise = np.zeros(len(x_test_noise[0,0,:]))

x_val_hfsep = data_hfsep[:8,:,-500:-100]
y_val_hfsep = np.ones(len(x_val_hfsep[0,0,:]))
x_val_noise = data_noise[:8,:,-500:-100]
y_val_noise = np.zeros(len(x_val_noise[0,0,:]))

x_train = np.expand_dims(np.append(x_train_hfsep, x_train_noise, axis=2), axis=3)
y_train = np.concatenate((y_train_hfsep, y_train_noise), axis=0)
x_val = np.expand_dims(np.append(x_val_hfsep, x_val_noise, axis=2), axis=3)
y_val = np.concatenate((y_val_hfsep, y_val_noise), axis=0)
x_test = np.expand_dims(np.append(x_test_hfsep, x_test_noise, axis=2), axis=3)
y_test = np.concatenate((y_test_hfsep, y_test_noise), axis=0)

## Swap axes so that the samples are in axis 0, then channels, time and 1
# print(x_train.shape)
x_train = np.swapaxes(np.swapaxes(x_train, 2, 0), 2, 1)
x_val = np.swapaxes(np.swapaxes(x_val, 2, 0), 2, 1)
x_test = np.swapaxes(np.swapaxes(x_test, 2, 0), 2, 1)
# print(x_train.shape)
# ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(y_train.shape)

## Train and save the model
max_iterations_for_training = 1000
epochs_per_loop_iteration = 25

for i in range(int(max_iterations_for_training / epochs_per_loop_iteration)):
	history = eeg_net.fit(x=x_train, y=y_train, epochs=epochs_per_loop_iteration, validation_data=(x_val, y_val))
	loss = history.history.get('loss')[-1]
	if loss <= 0.25:
		print('Stopped in epoch %d due to reaching loss of: %d' % (i, loss))
		print('Code to save the model and its history would go here. Instead, print the accuracy [%f] and the initial_learning_rate: %f' % (history.history.get('accuracy')[-1], initial_learning_rate))
		eeg_net.save('/home/christoph/Desktop/Beginning_September_Work/scripts/eegnet_baseline_prepped_data')
		break
