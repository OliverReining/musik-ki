import numpy as np
from data_preparation import process_midi_files
from model_preparation import prepare_training_data
from model import create_model

def train_model(midi_directory, sequence_length=100, epochs=50, batch_size=64):
    # Verarbeiten der MIDI-Dateien
    notes = process_midi_files(midi_directory)
    
    # Vorbereiten der Trainingsdaten
    X, y, note_to_int = prepare_training_data(notes, sequence_length)
    
    # Erstellen des Modells
    input_shape = (X.shape[1], X.shape[2])
    num_classes = len(note_to_int)
    model = create_model(input_shape, num_classes)
    
    # Trainieren des Modells
    model.fit(X, y, epochs=epochs, batch_size=batch_size)
    
    # Speichern des Modells und der Noten-Mapping
    model.save('../models/music_model.h5')
    with open('../models/note_to_int.npy', 'wb') as f:
        np.save(f, note_to_int)
    
    print("Training abgeschlossen und Modell gespeichert.")
    
if __name__ == "__main__":
    midi_directory = '../data'
    train_model(midi_directory)
