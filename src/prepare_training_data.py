import numpy as np
from tensorflow.keras.utils import to_categorical

def prepare_training_data(notes, sequence_length=100):
    """Bereitet die Trainingsdaten aus den Noten vor."""
    unique_notes = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(unique_notes)}

    X = []
    y = []
    for i in range(0, len(notes) - sequence_length):
        input_seq = notes[i:i + sequence_length]
        output_note = notes[i + sequence_length]
        X.append([note_to_int[note] for note in input_seq])
        y.append(note_to_int[output_note])

    X = np.reshape(X, (len(X), sequence_length, 1))
    X = X / float(len(unique_notes))
    y = to_categorical(y, num_classes=len(unique_notes))
    
    return X, y, note_to_int
