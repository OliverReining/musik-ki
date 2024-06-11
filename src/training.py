import numpy as np
import tensorflow as tf
from data_processing import midi_to_notes, preprocess_notes
from model import build_model

def load_data(midi_files, seq_length=50):
    all_notes = []
    for file in midi_files:
        notes = midi_to_notes(file)
        all_notes.extend(preprocess_notes(notes, seq_length))
    X_train = np.array([seq[0] for seq in all_notes])
    y_train = np.array([seq[1] for seq in all_notes])
    return X_train, y_train

def train_model(midi_files, seq_length=50, epochs=50, batch_size=64):
    X_train, y_train = load_data(midi_files, seq_length)
    model = build_model((X_train.shape[1], X_train.shape[2]))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=128)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save('models/music_model.h5')

if __name__ == "__main__":
    midi_files = ['path/to/your/midi/files']  # Hier musst du deine MIDI-Dateien angeben
    train_model(midi_files)
