from __future__ import print_function 
from numpy import load                                                                                                                                              
import numpy as np                                                                                                                                                  
from os import listdir 
import numpy as np 
from numpy import load 
import sys 

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

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
 
import scipy.io 

from pyts.image import RecurrencePlot

def normalize_z(to_be_normalized): 
	return (to_be_normalized - np.mean(to_be_normalized)) / (np.std(to_be_normalized))

rand_stat = 42

def loo_data_generator_train(id_to_leave_out): 
	for k in [x for x in range(1, 11) if x != id_to_leave_out]: 
		print('Now training on k00%d!' % k) 

		noise_k00X = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % k)
		hfsep_k00X = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % k)

		znorm_noise = normalize_z(noise_k00X[:8, :, :])
		znorm_hfsep = normalize_z(hfsep_k00X[:8, :, :])

		X_train = np.swapaxes(np.concatenate((znorm_hfsep, znorm_noise), axis=2), 2, 0)
		y_train = np.concatenate((np.ones(znorm_hfsep.shape[-1], dtype='int8'), np.zeros(znorm_noise.shape[-1], dtype='int8')), axis=0)

		# Remove dimensions to assure dividability:
		X_train = X_train[:(32 * (X_train.shape[0] // 32))]
		y_train = y_train[:(32 * (y_train.shape[0] // 32))]

		X_train, y_train = shuffle(X_train, y_train, random_state=rand_stat)

		# Our vectorized labels
		y_train = y_train.reshape((-1,1))

		print('X_train shape')
		print(X_train.shape)
		print('y_train shape')
		print(y_train.shape)
		
		for i in range(X_train.shape[0]):
			yield (rp.fit_transform(X_train[i].T), y_train[i])


def loo_data_generator_test(id_to_leave_out): 
	print('Now evaluating on k00%d!' % id_to_leave_out)

	noise_k00X = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % id_to_leave_out)
	hfsep_k00X = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % id_to_leave_out)

	znorm_noise = normalize_z(noise_k00X[:8, :, :])
	znorm_hfsep = normalize_z(hfsep_k00X[:8, :, :])

	X_test = np.swapaxes(np.concatenate((znorm_hfsep, znorm_noise), axis=2), 2, 0)
	y_test = np.concatenate((np.ones(znorm_hfsep.shape[-1], dtype='int8'), np.zeros(znorm_noise.shape[-1], dtype='int8')), axis=0)

	print('X_test shape')
	print(X_test.shape)

	# Remove dimensions to assure dividability:
	X_test = X_test[:(32 * (X_test.shape[0] // 32))]
	y_test = y_test[:(32 * (y_test.shape[0] // 32))]

	X_train, y_train = shuffle(X_test, y_test, random_state=rand_stat)

	# Our vectorized labels
	y_test = y_test.reshape((-1,1))

	for i in range(X_test.shape[0]):
		yield (rp.fit_transform(X_test[i].T))


def get_labels_from_loo_data_generator_test(id_to_leave_out): 
	print('Now evaluating on k00%d!' % id_to_leave_out)

	noise_k00X = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_noise.npy' % id_to_leave_out)
	hfsep_k00X = load('/media/christoph/Volume/Masterthesis/final_prepped_data/k00%d/epoched_intrplt_filt_500_900_kx_hfsep.npy' % id_to_leave_out)

	znorm_noise = normalize_z(noise_k00X[:8, :, :])
	znorm_hfsep = normalize_z(hfsep_k00X[:8, :, :])

	X_test = np.swapaxes(np.concatenate((znorm_hfsep, znorm_noise), axis=2), 2, 0)
	y_test = np.concatenate((np.ones(znorm_hfsep.shape[-1], dtype='int8'), np.zeros(znorm_noise.shape[-1], dtype='int8')), axis=0)

	print('X_test shape')
	print(X_test.shape)

	# Remove dimensions to assure dividability:
	X_test = X_test[:(32 * (X_test.shape[0] // 32))]
	y_test = y_test[:(32 * (y_test.shape[0] // 32))]

	_, y_train = shuffle(X_test, y_test, random_state=rand_stat)

	# Our vectorized labels
	y_test = y_test.reshape((-1,1))

	return y_test


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


#### #### #### #### ####
#### Model Definit. ####
#### #### #### #### ####

class TemporalSplit(tf.keras.layers.Layer):
	"""Split the input tensor into 8 tensors along the spatial dimension."""

	def call(self, inputs):
		# Expect the input to be 3D and mask to be 2D, split the input tensor into 8
		# subtensors along the time axis (axis 1).
		print("INSIDE OF SPLIT")
		print(inputs)
		print(inputs.shape)
		return tf.split(inputs, 8, axis=1)

	def compute_mask(self, inputs, mask=None):
		# Also split the mask into 8 if it presents.
		if mask is None:
			return None
		return tf.split(mask, 8, axis=1)


def build_model_after_deeprnn(in_shape=(250, 250, 8), grad_clip=110, imsize = 32, n_colors = 3, n_timewin = 8, optimizer='adam', learning_rate=0.001, clip_vals=False):

	lr = 0.001

	input_layer = tf.keras.Input(shape=in_shape)
	split_layer_out = TemporalSplit()(input_layer)
	print(split_layer_out)

	convnets = []
	# Build parallel CNNs with shared weights
	for i in range(n_timewin):
		reshape_in = tf.keras.layers.Reshape((250, 250, 1))(split_layer_out[i])
		conv_0 = tf.keras.layers.Conv2D(16, 3, padding='same')(reshape_in)
		conv_1 = tf.keras.layers.Conv2D(16, 3, padding='same')(conv_0)
		conv_2 = tf.keras.layers.Conv2D(16, 3, padding='same')(conv_1)
		conv_3 = tf.keras.layers.Conv2D(16, 3, padding='same')(conv_2)
		max_0 = tf.keras.layers.MaxPool2D()(conv_3)
		conv_4 = tf.keras.layers.Conv2D(32, 3, padding='same')(max_0)
		conv_5 = tf.keras.layers.Conv2D(32, 3, padding='same')(conv_4)
		max_1 = tf.keras.layers.MaxPool2D()(conv_5)
		conv_6 = tf.keras.layers.Conv2D(64, 3, padding='same')(max_1)
		max_2 = tf.keras.layers.MaxPool2D()(conv_6)
		flat = tf.keras.layers.Flatten()(max_2)
		convnets.append(flat)

	# Now concatenate the parallel CNNs to one model
	concatted = tf.keras.layers.concatenate(convnets)

	# at this point convnets shape is [numTimeWin][n_samples, features]
	# we want the shape to be [n_samples, features, numTimeWin]
	# Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
	num_features = 61504
	reshaped = tf.keras.layers.Reshape((n_timewin, num_features))(concatted)
	lstm = tf.keras.layers.LSTM(128)(reshaped)
	den_0 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(lstm)
	drop_0 = tf.keras.layers.Dropout(0.5)(den_0)
	den_1 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(drop_0)

	model = tf.keras.Model(inputs=input_layer, outputs=den_1)
	print(model.summary())

	METRICS = [
		tf.keras.metrics.TruePositives(),
		tf.keras.metrics.FalsePositives(),
		tf.keras.metrics.TrueNegatives(),
		tf.keras.metrics.FalseNegatives(), 
		tf.keras.metrics.BinaryAccuracy(),
		tf.keras.metrics.Precision(),
		tf.keras.metrics.Recall(),
		tf.keras.metrics.AUC(),
		tf.keras.metrics.MeanAbsoluteError()
	]

	if optimizer is 'adam':
		print("Compiling with ADAM")
		if clip_vals:
			print("Doing it with clipvals")
			model.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0), 
				loss=tf.keras.losses.BinaryCrossentropy(),
				metrics=METRICS)
		else:
			model.compile(optimizer=tf.keras.optimizers.Adam(lr), 
				loss=tf.keras.losses.BinaryCrossentropy(),
				metrics=METRICS)
	else:
		print("Compiling with SGD")
		if clip_vals:
			print("Doing it with clipvals")
			model.compile(optimizer=tf.keras.optimizers.SGD(lr, momentum=0.9, clipvalue=5.0), 
				loss=tf.keras.losses.BinaryCrossentropy(),
				metrics=METRICS)
		else:
			model.compile(optimizer=tf.keras.optimizers.SGD(lr, momentum=0.9), 
				loss=tf.keras.losses.BinaryCrossentropy(),
				metrics=METRICS)

	return model

#### #### #### #### ####
#### Model Training ####
#### #### #### #### ####

for id_to_leave_out in range(1, 11):

	rp = RecurrencePlot()
	model = build_model_after_deeprnn(optimizer='sgd', learning_rate=0.001, clip_vals=True)

	history = model.fit(x=tf.data.Dataset.from_generator(lambda: loo_data_generator_train(id_to_leave_out), (tf.float32, tf.uint8), ([250, 250, 8], [None])).batch(32), epochs=25) 

	model.save('/media/christoph/Volume/Masterthesis/recurrence_plot_training/loo/models_trained_and_saved_test_place/rec_deep_rnn_w_sgd_lr_0001_clip_True_loo_k00%d_left_out' % id_to_leave_out)
	np.save('/media/christoph/Volume/Masterthesis/recurrence_plot_training/loo/histories_of_models_trained_and_saved_test_place/rec_deep_rnn_w_sgd_lr_0001_clip_True_loo_k00%d_left_out' % id_to_leave_out, history.history)

	predictions = np.abs(np.rint(model.predict(x=tf.data.Dataset.from_generator(lambda: loo_data_generator_test(id_to_leave_out), (tf.float32), ([250, 250, 8])).batch(32))))
	confusion_matrix_for_model = confusion_matrix(get_labels_from_loo_data_generator_test(id_to_leave_out), predictions)
	tn, fp, fn, tp = confusion_matrix_for_model.ravel()
	np.save('/media/christoph/Volume/Masterthesis/recurrence_plot_training/loo/histories_of_models_trained_and_saved_test_place_confusion_matrices/rec_deep_rnn_w_sgd_lr_0001_clip_True_loo_k00%d_left_out_tn_%d_fp_%d_fn_%d_tp_%d_confusion_matrix' % (id_to_leave_out, tn, fp, fn, tp), confusion_matrix_for_model)

	metrics = metrics_for_conf_mat(tn, fp, fn, tp)
	np.save('/media/christoph/Volume/Masterthesis/recurrence_plot_training/loo/histories_of_models_trained_and_saved_test_place_confusion_matrices/rec_deep_rnn_w_sgd_lr_0001_clip_True_loo_k00%d_left_out_tn_%d_fp_%d_fn_%d_tp_%d_metrics' % (id_to_leave_out, tn, fp, fn, tp), metrics)
