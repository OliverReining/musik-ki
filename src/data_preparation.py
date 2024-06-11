import os
from music21 import converter, instrument, note, chord

def midi_to_notes(midi_file):
    """Konvertiert eine MIDI-Datei in eine Liste von Noten und Akkorden."""
    midi = converter.parse(midi_file)
    notes = []
    for part in midi.parts:
        # Nur die Instrumente betrachten, die nicht Gesang sind
        if not any(isinstance(instr, instrument.Vocal) for instr in part.getElementsByClass(instrument.Instrument)):
            for element in part.recurse():
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def process_midi_files(directory):
    """Verarbeitet alle MIDI-Dateien in einem Verzeichnis."""
    all_notes = []
    for midi_file in os.listdir(directory):
        if midi_file.endswith(".mid"):
            midi_path = os.path.join(directory, midi_file)
            notes = midi_to_notes(midi_path)
            all_notes.extend(notes)
    return all_notes
