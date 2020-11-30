from os import listdir
import numpy as np
from numpy import load
import sys

X_Test = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_test.npy', allow_pickle=True)[:,29:-26,40:-82,:] / 255.0
X_Train = load('/media/christoph/Volume/Masterthesis/spectograms_concatenated_as_modality_in_npy_format/ccar_filt_intrplt_filt_500_900_kx_k003_final_train.npy', allow_pickle=True)[:,29:-26,40:-82,:] / 255.0
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

print(sys.argv)

## Train and save the model
max_epochs = 25

i = int(sys.argv[1])
j = float(sys.argv[2])
k = int(sys.argv[3])
l = str(sys.argv[4])
m = int(sys.argv[5])

model = keras.Sequential()
model.add(layers.Conv2D(16 * i, (5, 5), input_shape=(185, 198, 4), activation='relu', padding='same'))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(16 * i, (k, k), activation=l, padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32 * i, (3, 3), activation='relu', padding='same'))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(16 * i, (3, 3), activation=l, padding='same'))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.1))
model.add(layers.Flatten())
model.add(layers.Dense(32 * i, activation=l, kernel_constraint=tf.keras.constraints.MaxNorm(3)))
model.add(layers.Dense(16 * i, activation='relu', kernel_constraint=tf.keras.constraints.MaxNorm(3)))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), 
    optimizer=tf.keras.optimizers.SGD(learning_rate=j, momentum=0.9), 
    metrics=['accuracy'])

history = model.fit(
    x=X_Train[:5000], 
    y=y_train[:5000], 
    epochs=max_epochs, 
    batch_size=m,
    validation_data=(X_Test[:2000], y_test[:2000]))

#    callbacks =[earlystopping]

model.save('/media/christoph/Volume/Masterthesis/models_trained_and_saved_test_place/simple_2d_cnn_after_lenet_%d_%f_%d_%s_%d' % (i, j, k, l, m))
np.save('/media/christoph/Volume/Masterthesis/histories_of_models_trained_and_saved_test_place/simple_2d_cnn_after_lenet_%d_%f_%d_%s_%d' % (i, j, k, l, m), history.history)