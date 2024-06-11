# src/data_processing.py

import pretty_midi
import numpy as np
import os

def midi_to_notes(midi_file):
    pm = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append([note.start, note.end, note.pitch])
    return np.array(sorted(notes, key=lambda x: x[0]))

def preprocess_notes(notes, seq_length=50):
    sequences = []
    for i in range(len(notes) - seq_length):
        seq_in = notes[i:i + seq_length]
        seq_out = notes[i + seq_length]
        sequences.append((seq_in, seq_out))
    return sequences

def load_midi_files(data_dir):
    midi_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mid')]
    all_notes = []
    for file in midi_files:
        notes = midi_to_notes(file)
        all_notes.extend(preprocess_notes(notes))
    return all_notes

def load_user_midi_file(midi_file):
    notes = midi_to_notes(midi_file)
    return preprocess_notes(notes)
