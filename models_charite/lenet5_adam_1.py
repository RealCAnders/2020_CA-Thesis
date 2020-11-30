from os import listdir
import numpy as np
from numpy import load

X_Test = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_test.npy', allow_pickle=True)
X_Train = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_train.npy', allow_pickle=True)
y_test = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_test_labels.npy', allow_pickle=True)
y_train = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_train_labels.npy', allow_pickle=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

## Train and save the model
max_epochs = 500
earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_binary_crossentropy",  
            mode ="min", patience = 10,  
            restore_best_weights = True)
earlystopping_max = tf.keras.callbacks.EarlyStopping(monitor ="val_binary_crossentropy",  
            mode ="max", patience = 10,  
            restore_best_weights = True)

model = keras.Sequential()

model.add(layers.Conv2D(filters=60, kernel_size=(3, 3), input_shape=(240,320,4)))
model.add(layers.LeakyReLU(alpha=0.2))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=120, kernel_size=(3, 3)))
model.add(layers.LeakyReLU(alpha=0.2))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=1, activation='sigmoid'))

model.compile(
	loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
	optimizer=tf.keras.optimizers.Adam(learning_rate=1.0), 
	metrics=[
		tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'), 
		'accuracy', 
		tf.metrics.BinaryAccuracy(threshold=0.0)])

history = model.fit(
    x=X_Train[:5000], 
    y=y_train[:5000], 
    epochs=max_epochs, 
    batch_size=64,
    validation_data=(X_Test[:2000], y_test[:2000]),
    callbacks =[earlystopping, earlystopping_max])

model.save('/media/christoph/Volume/Masterthesis/models_trained_and_saved_test_place/le_net_5_adam_1')
np.save('/media/christoph/Volume/Masterthesis/histories_of_models_trained_and_saved_test_place/le_net_5_adam_1', history.history)
