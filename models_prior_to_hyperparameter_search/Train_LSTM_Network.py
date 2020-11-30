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
    
    lstm_one = layers.LSTM(8)(chan)
    out = tf.keras.layers.Activation('sigmoid')(lstm_one)
    
    channel_model = tf.keras.Model(inputs=chan, outputs=out)
    channel_models.append(channel_model)

# concatente the 1D-channels to a MC-Model
combined_model = tf.keras.layers.concatenate([tf.keras.backend.expand_dims(channl.output, axis=1) for channl in channel_models], axis=1)
conv_one = layers.Conv1D(8, 2, input_shape=(8, 8))(combined_model)
flatten = layers.Flatten()(conv_one)
dense = layers.Dense(1)(flatten)
sigmoid = layers.Activation('sigmoid', name = 'sigmoid')(dense)

model = keras.Model(inputs=inputs, outputs=sigmoid)

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

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),  # Optimizer
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS)

data_hfsep_1 = load("/home/christoph/Desktop/Beginning_September_Work/Prepped_Data/OutliersRejected/k001_500_900_int_70_70_hfsep_100_300.npy")
print(data_hfsep_1.shape)
data_noise_1 = load("/home/christoph/Desktop/Beginning_September_Work/Prepped_Data/OutliersRejected/k001_500_900_int_70_70_noise_100_300.npy")
print(data_noise_1.shape)
data_hfsep_2 = load("/home/christoph/Desktop/Beginning_September_Work/Prepped_Data/OutliersRejected/k002_500_900_int_70_70_hfsep_100_300.npy")
print(data_hfsep_2.shape)
data_noise_2 = load("/home/christoph/Desktop/Beginning_September_Work/Prepped_Data/OutliersRejected/k002_500_900_int_70_70_noise_100_300.npy")
print(data_noise_2.shape)
data_hfsep_3 = load("/home/christoph/Desktop/Beginning_September_Work/Prepped_Data/OutliersRejected/k003_500_900_int_70_70_hfsep_100_300.npy")
print(data_hfsep_3.shape)
data_noise_3 = load("/home/christoph/Desktop/Beginning_September_Work/Prepped_Data/OutliersRejected/k003_500_900_int_70_70_noise_100_300.npy")
print(data_noise_3.shape)
data_hfsep_4 = load("/home/christoph/Desktop/Beginning_September_Work/Prepped_Data/OutliersRejected/k004_500_900_int_70_70_hfsep_100_300.npy")
print(data_hfsep_4.shape)
data_noise_4 = load("/home/christoph/Desktop/Beginning_September_Work/Prepped_Data/OutliersRejected/k004_500_900_int_70_70_noise_100_300.npy")
print(data_noise_4.shape)

## Combine data that has shape: channels x time_datapoint x sample
x_train_hfsep_1 = data_hfsep_1[:8,:,:-500]
y_train_hfsep_1 = np.ones(len(x_train_hfsep_1[0,0,:]), dtype=np.int8)
x_train_noise_1 = data_noise_1[:8,:,:-500]
y_train_noise_1 = np.zeros(len(x_train_noise_1[0,0,:]), dtype=np.int8)

x_test_hfsep_1 = data_hfsep_1[:8,:,-100:]
y_test_hfsep_1 = np.ones(len(x_test_hfsep_1[0,0,:]), dtype=np.int8)
x_test_noise_1 = data_noise_1[:8,:,-100:]
y_test_noise_1 = np.zeros(len(x_test_noise_1[0,0,:]), dtype=np.int8)

x_val_hfsep_1 = data_hfsep_1[:8,:,-500:-100]
y_val_hfsep_1 = np.ones(len(x_val_hfsep_1[0,0,:]), dtype=np.int8)
x_val_noise_1 = data_noise_1[:8,:,-500:-100]
y_val_noise_1 = np.zeros(len(x_val_noise_1[0,0,:]), dtype=np.int8)

x_train_hfsep_2 = data_hfsep_2[:8,:,:-500]
y_train_hfsep_2 = np.ones(len(x_train_hfsep_2[0,0,:]), dtype=np.int8)
x_train_noise_2 = data_noise_2[:8,:,:-500]
y_train_noise_2 = np.zeros(len(x_train_noise_2[0,0,:]), dtype=np.int8)

x_test_hfsep_2 = data_hfsep_2[:8,:,-100:]
y_test_hfsep_2 = np.ones(len(x_test_hfsep_2[0,0,:]), dtype=np.int8)
x_test_noise_2 = data_noise_2[:8,:,-100:]
y_test_noise_2 = np.zeros(len(x_test_noise_2[0,0,:]), dtype=np.int8)

x_val_hfsep_2 = data_hfsep_2[:8,:,-500:-100]
y_val_hfsep_2 = np.ones(len(x_val_hfsep_2[0,0,:]), dtype=np.int8)
x_val_noise_2 = data_noise_2[:8,:,-500:-100]
y_val_noise_2 = np.zeros(len(x_val_noise_2[0,0,:]), dtype=np.int8)

x_train_hfsep_3 = data_hfsep_3[:8,:,:-500]
y_train_hfsep_3 = np.ones(len(x_train_hfsep_3[0,0,:]), dtype=np.int8)
x_train_noise_3 = data_noise_3[:8,:,:-500]
y_train_noise_3 = np.zeros(len(x_train_noise_3[0,0,:]), dtype=np.int8)

x_test_hfsep_3 = data_hfsep_3[:8,:,-100:]
y_test_hfsep_3 = np.ones(len(x_test_hfsep_3[0,0,:]), dtype=np.int8)
x_test_noise_3 = data_noise_3[:8,:,-100:]
y_test_noise_3 = np.zeros(len(x_test_noise_3[0,0,:]), dtype=np.int8)

x_val_hfsep_3 = data_hfsep_3[:8,:,-500:-100]
y_val_hfsep_3 = np.ones(len(x_val_hfsep_3[0,0,:]), dtype=np.int8)
x_val_noise_3 = data_noise_3[:8,:,-500:-100]
y_val_noise_3 = np.zeros(len(x_val_noise_3[0,0,:]), dtype=np.int8)

x_train_hfsep_4 = data_hfsep_4[:8,:,:-500]
y_train_hfsep_4 = np.ones(len(x_train_hfsep_4[0,0,:]), dtype=np.int8)
x_train_noise_4 = data_noise_4[:8,:,:-500]
y_train_noise_4 = np.zeros(len(x_train_noise_4[0,0,:]), dtype=np.int8)

x_test_hfsep_4 = data_hfsep_4[:8,:,-100:]
y_test_hfsep_4 = np.ones(len(x_test_hfsep_4[0,0,:]), dtype=np.int8)
x_test_noise_4 = data_noise_4[:8,:,-100:]
y_test_noise_4 = np.zeros(len(x_test_noise_4[0,0,:]), dtype=np.int8)

x_val_hfsep_4 = data_hfsep_4[:8,:,-500:-100]
y_val_hfsep_4 = np.ones(len(x_val_hfsep_4[0,0,:]), dtype=np.int8)
x_val_noise_4 = data_noise_4[:8,:,-500:-100]
y_val_noise_4 = np.zeros(len(x_val_noise_4[0,0,:]), dtype=np.int8)

x_train = np.concatenate((x_train_hfsep_1, x_train_noise_1, x_train_hfsep_2, x_train_noise_2, x_train_hfsep_3, x_train_noise_3, x_train_hfsep_4, x_train_noise_4), axis=2)
y_train = np.concatenate((y_train_hfsep_1, y_train_noise_1, y_train_hfsep_2, y_train_noise_2, y_train_hfsep_3, y_train_noise_3, y_train_hfsep_4, y_train_noise_4), axis=0)
x_val = np.concatenate((x_val_hfsep_1, x_val_noise_1, x_val_hfsep_2, x_val_noise_2, x_val_hfsep_3, x_val_noise_3, x_val_hfsep_4, x_val_noise_4), axis=2)
y_val = np.concatenate((y_val_hfsep_1, y_val_noise_1, y_val_hfsep_2, y_val_noise_2, y_val_hfsep_3, y_val_noise_3, y_val_hfsep_4, y_val_noise_4), axis=0)
x_test = np.concatenate((x_test_hfsep_1, x_test_noise_1, x_test_hfsep_2, x_test_noise_2, x_test_hfsep_3, x_test_noise_3, x_test_hfsep_4, x_test_noise_4), axis=2)
y_test = np.concatenate((y_test_hfsep_1, y_test_noise_1, y_test_hfsep_2, y_test_noise_2, y_test_hfsep_3, y_test_noise_3, y_test_hfsep_4, y_test_noise_4), axis=0)

## Swap axes so that the samples are in axis 0, then channels, time and 1
print(x_train.shape)
x_train = np.swapaxes(np.expand_dims(np.expand_dims(np.swapaxes(x_train, 1, 2), axis=2), axis=3), 3, 4)
x_val = np.swapaxes(np.expand_dims(np.expand_dims(np.swapaxes(x_val, 1, 2), axis=2), axis=3), 3, 4)
x_test = np.swapaxes(np.expand_dims(np.expand_dims(np.swapaxes(x_test, 1, 2), axis=2), axis=3), 3, 4)
print(x_train.shape)
# ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

## Train and save the model
max_iterations_for_training = 1000
epochs_per_loop_iteration = 25

for i in range(int(max_iterations_for_training / epochs_per_loop_iteration)):
	history = model.fit(
        x=[np.squeeze(np.squeeze(x, axis=2), axis=0) for x in np.split(x_train, indices_or_sections=8, axis=0)],
        y=y_train, batch_size=128, epochs=epochs_per_loop_iteration, 
        validation_data=([np.squeeze(np.squeeze(x, axis=2), axis=0) for x in np.split(x_val, indices_or_sections=8, axis=0)], y_val))
	loss = history.history.get('loss')[-1]
	if loss <= 0.25:
		print('Stopped in epoch %d due to reaching loss of: %d' % (i, loss))
		print('Code to save the model and its history would go here. Instead, print the accuracy [%f] and the initial_learning_rate: %f' % (history.history.get('accuracy')[-1], initial_learning_rate))
		model.save('/home/christoph/Desktop/Beginning_September_Work/scripts/EEG_CNN_baseline_prepped_data')
		break