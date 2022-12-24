import tensorflow as tf
from keras.models import Sequential
from tensorflow import keras
from keras import layers



def BUILD_MODEL(X_train):
    
    model=Sequential()
    model.add(layers.SeparableConv1D(filters=24, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', input_shape=X_train.shape[1:]))
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.SeparableConv1D(filters=32, kernel_size=3, activation=tf.nn.leaky_relu, padding='same'))
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.SeparableConv1D(filters=64, kernel_size=3, activation=tf.nn.leaky_relu, padding='same'))
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.SeparableConv1D(filters=128, kernel_size=3, activation=tf.nn.leaky_relu, padding='same'))
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.SeparableConv1D(filters=160, kernel_size=3, activation=tf.nn.leaky_relu, padding='same'))
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.SeparableConv1D(filters=320, kernel_size=3, activation=tf.nn.leaky_relu, padding='same'))
    model.add(layers.MaxPooling1D(2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    model.summary()
    
    return model