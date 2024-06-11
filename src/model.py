import tensorflow as tf
from tensorflow.keras import layers

def build_model(input_shape):
    model = tf.keras.Sequential([
        layers.LSTM(512, input_shape=input_shape, return_sequences=True),
        layers.LSTM(512),
        layers.Dense(128, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model
