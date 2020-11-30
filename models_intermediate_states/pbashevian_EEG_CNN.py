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

## Some model parameters and model definition
# shape of the image (SHAPE x SHAPE)
shapex, shapey = 8, 200
# number of convolutional filters to use
nb_filters = 2
# level of pooling to perform (POOL x POOL)
nb_pool = 2
# level of convolution to perform (CONV x CONV)
nb_conv = 2
# amount of classes
nb_classes = 2

model = models.Sequential()

model.add(keras.Input(shape = (8, 200, 1)))
model.add(layers.Conv2D(filters=nb_filters, kernel_size=1, strides=(nb_conv, nb_conv), padding="valid"))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(filters=nb_filters, kernel_size=nb_filters, strides=(nb_conv, nb_conv)))
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

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.5),  # Optimizer
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
x_train = np.expand_dims(np.swapaxes(np.swapaxes(x_train, 2, 0), 2, 1), axis=3)
x_val = np.expand_dims(np.swapaxes(np.swapaxes(x_val, 2, 0), 2, 1), axis=3)
x_test = np.expand_dims(np.swapaxes(np.swapaxes(x_test, 2, 0), 2, 1), axis=3)
print(x_train.shape)
# ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

## Train and save the model
max_iterations_for_training = 1000
epochs_per_loop_iteration = 25

for i in range(int(max_iterations_for_training / epochs_per_loop_iteration)):
	history = model.fit(x=x_train, y=y_train, epochs=epochs_per_loop_iteration, validation_data=(x_val, y_val))
	loss = history.history.get('loss')[-1]
	if loss <= 0.25:
		print('Stopped in epoch %d due to reaching loss of: %d' % (i, loss))
		print('Code to save the model and its history would go here. Instead, print the accuracy [%f] and the initial_learning_rate: %f' % (history.history.get('accuracy')[-1], initial_learning_rate))
		model.save('/home/christoph/Desktop/Beginning_September_Work/scripts/EEG_CNN_baseline_prepped_data')
		break