import pretty_midi
import numpy as np

def midi_to_notes(midi_file):
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = []
    for note in instrument.notes:
        notes.append([note.start, note.end, note.pitch])
    return np.array(notes)

def preprocess_notes(notes, seq_length=50):
    sequences = []
    for i in range(len(notes) - seq_length):
        seq_in = notes[i:i + seq_length]
        seq_out = notes[i + seq_length]
        sequences.append((seq_in, seq_out))
    return sequences
