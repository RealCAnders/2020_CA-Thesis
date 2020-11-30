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
def MC_CNN():
    
    lr = 0.005
    classification_threshold = 0.5
    time_lenght = 200

    # Define the respective channels for the electrodes signals
    in_fz = tf.keras.Input((time_lenght, 1), name='FZ')
    in_f3 = tf.keras.Input((time_lenght, 1), name='F3')
    in_fc5 = tf.keras.Input((time_lenght, 1), name='FC5')
    in_cz = tf.keras.Input((time_lenght, 1), name='CZ')
    in_c3 = tf.keras.Input((time_lenght, 1), name='C3')
    in_cp5 = tf.keras.Input((time_lenght, 1), name='CP5')
    in_t7 = tf.keras.Input((time_lenght, 1), name='T7')
    in_cp1 = tf.keras.Input((time_lenght, 1), name='CP1')
    
    inputs = [in_fz, in_f3, in_fc5, in_cz, in_c3, in_cp5, in_t7, in_cp1]
    channel_models = []
    
    # Define three 1D-Convolutions with Pooling after each input_channel
    for chan in inputs:
        
        c_conv = tf.keras.layers.Conv1D(32, 3, use_bias=False)(chan)
        c_pool = tf.keras.layers.AveragePooling1D(pool_size=2)(c_conv)
        c_conv = tf.keras.layers.Conv1D(16, 4, use_bias=False)(c_pool)
        c_pool = tf.keras.layers.MaxPool1D(pool_size=2)(c_conv)
        out = tf.keras.layers.Activation('sigmoid')(c_pool)
        
        channel_model = tf.keras.Model(inputs=chan, outputs=out)
        channel_models.append(channel_model)
    
    # concatente the 1D-channels to a MC-Model
    combined_model = tf.keras.layers.concatenate(
        [tf.keras.backend.expand_dims(channl.output, axis=1) for channl in channel_models]
        , axis=1
    )
    
    # further define the MC-Model using Conv2D, Dense, Pooling and Flattening
    conv = tf.keras.layers.Conv2D(1, (4, 5))(combined_model)
    den = tf.keras.layers.Dense(120)(conv)
    conv = tf.keras.layers.Conv2D(1, (5, 15))(den)
    den = tf.keras.layers.Dense(40)(conv)
    conv = tf.keras.layers.Conv2D(20, (1, 10))(den)
    avg = tf.keras.layers.AveragePooling2D((1, 8))(conv)
    flat = tf.keras.layers.Flatten()(avg)
    den = tf.keras.layers.Dense(10)(flat)
    out = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(den)
    
    # Build the final model, show a summary and compile it w. metrics
    mc_cnn = tf.keras.Model(inputs=inputs, outputs=out)

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

    mc_cnn.compile(optimizer=tf.keras.optimizers.Adam(lr), 
                   loss=tf.keras.losses.BinaryCrossentropy(),
                   metrics=METRICS)
    
    return mc_cnn

mc_cnn = MC_CNN()

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
print(x_train.shape)
x_train = np.expand_dims(np.swapaxes(x_train, 1, 2), axis=2)
x_val = np.expand_dims(np.swapaxes(x_val, 1, 2), axis=2)
x_test = np.expand_dims(np.swapaxes(x_test, 1, 2), axis=2)
print(x_train.shape)
# ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

#for x in np.split(x_train, indices_or_sections=8, axis=0):
#	print(x.shape)

#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(y_train.shape)

## Train and save the model
max_iterations_for_training = 1000
epochs_per_loop_iteration = 25

for i in range(int(max_iterations_for_training / epochs_per_loop_iteration)):
	history = mc_cnn.fit(
		x=[np.squeeze(np.squeeze(x, axis=2), axis=0) for x in np.split(x_train, indices_or_sections=8, axis=0)], 
		y=y_train, 
		epochs=epochs_per_loop_iteration, 
		validation_data=([np.squeeze(np.squeeze(x, axis=2), axis=0) for x in np.split(x_val, indices_or_sections=8, axis=0)], y_val))
	loss = history.history.get('loss')[-1]
	if loss <= 0.25:
		print('Stopped in epoch %d due to reaching loss of: %d' % (i, loss))
		print('Code to save the model and its history would go here. Instead, print the accuracy [%f] and the initial_learning_rate: %f' % (history.history.get('accuracy')[-1], initial_learning_rate))
		mc_cnn.save('/home/christoph/Desktop/Beginning_September_Work/scripts/mc_cnn_baseline_prepped_data')
		break
