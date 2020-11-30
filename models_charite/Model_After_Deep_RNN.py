def build_model_after_deeprnn(in_shape=(185, 198, 4), grad_clip=110, imsize = 32, n_colors = 3, n_timewin = 3):
    
    lr = 0.001

    input_layer = tf.keras.Input(shape=in_shape)

    convnets = []
    # Build parallel CNNs with shared weights
    for i in range(n_timewin):
        conv_0 = tf.keras.layers.Conv2D(32 * (2 ** 2), 3, padding='same')(input_layer)
        conv_1 = tf.keras.layers.Conv2D(32 * (2 ** 2), 3, padding='same')(conv_0)
        conv_2 = tf.keras.layers.Conv2D(32 * (2 ** 2), 3, padding='same')(conv_1)
        max_0 = tf.keras.layers.MaxPool2D()(conv_2)
        conv_4 = tf.keras.layers.Conv2D(32 * (2 ** 1), 3, padding='same')(max_0)
        conv_5 = tf.keras.layers.Conv2D(32 * (2 ** 1), 3, padding='same')(conv_4)
        max_1 = tf.keras.layers.MaxPool2D()(conv_5)
        conv_6 = tf.keras.layers.Conv2D(32, 3, padding='same')(max_1)
        max_2 = tf.keras.layers.MaxPool2D()(conv_6)
        flat = tf.keras.layers.Flatten()(max_2)
        convnets.append(flat)

    # Now concatenate the parallel CNNs to one model
    concatted = tf.keras.layers.concatenate(convnets)

    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    num_features = 17664
    reshaped = tf.keras.layers.Reshape((n_timewin, num_features))(concatted)
    lstm = tf.keras.layers.LSTM(128)(reshaped)
    den_0 = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(lstm)
    drop_0 = tf.keras.layers.Dropout(0.5)(den_0)
    den_1 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(drop_0)

    model = tf.keras.Model(inputs=input_layer, outputs=den_1)
    print(model.summary())

#-#-#    OLD: concatente the 1D-channels to a MC-Model
#-#-#    combined_model = tf.keras.layers.concatenate(
#-#-#        [tf.keras.backend.expand_dims(channl.output, axis=1) for channl in channel_models]
#-#-#        , axis=1
#-#-#    )
    
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

    model.compile(optimizer=tf.keras.optimizers.Adam(lr, clipvalue=0.5), 
                   loss=tf.keras.losses.BinaryCrossentropy(),
                   metrics=METRICS)
    
    return model