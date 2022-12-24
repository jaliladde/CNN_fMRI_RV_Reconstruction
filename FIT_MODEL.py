import tensorflow as tf
from tensorflow import keras
from R2 import R2
from keras.callbacks import EarlyStopping, ModelCheckpoint


def FIT_MODEL(model, train_inputs, train_targets, valid_inputs, valid_targets, num_epochs, BatchSize, main_path):
    save_path = main_path + 'saved_data/'
    optimizer = tf.keras.optimizers.Adam()

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', R2])
    model.summary()

    earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min')
    model_save = ModelCheckpoint(save_path + 'roi90cnn.hdf5', save_best_only=True, monitor='val_R2', mode='min')

    #history = model.fit(train_inputs, train_targets, epochs=num_epochs, validation_split=valid_ratio, batch_size=BatchSize, callbacks=[earlyStopping, model_save])
    history = model.fit(train_inputs, train_targets, epochs=num_epochs, 
                        validation_data=(valid_inputs, valid_targets), 
                        batch_size=BatchSize, callbacks=[earlyStopping, model_save])
    
    return history