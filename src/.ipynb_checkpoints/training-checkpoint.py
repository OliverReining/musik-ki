# src/training.py

import numpy as np
import tensorflow as tf
from data_processing import load_midi_files, load_user_midi_file
from model import build_model

def load_data(data_dir, seq_length=50):
    all_notes = load_midi_files(data_dir)
    X_train = np.array([seq[0] for seq in all_notes])
    y_train = np.array([seq[1] for seq in all_notes])
    return X_train, y_train

def train_model(data_dir, seq_length=50, epochs=50, batch_size=64):
    X_train, y_train = load_data(data_dir, seq_length)
    model = build_model((X_train.shape[1], X_train.shape[2]))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=128)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save('models/music_model.h5')
    return model

def train_user_model(midi_file, seq_length=50, epochs=50, batch_size=64):
    user_notes = load_user_midi_file(midi_file)
    X_train = np.array([seq[0] for seq in user_notes])
    y_train = np.array([seq[1] for seq in user_notes])
    model = build_model((X_train.shape[1], X_train.shape[2]))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=128)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save('models/user_music_model.h5')
    return model

if __name__ == "__main__":
    data_dir = '../data'  # Pfad zum Verzeichnis mit den MIDI-Dateien
    model = train_model(data_dir)
