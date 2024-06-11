import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape):
    model = models.Sequential([
        layers.LSTM(512, input_shape=input_shape, return_sequences=True),
        layers.LSTM(512),
        layers.Dense(128, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def generate_music(model, start_sequence, num_notes=100):
    generated = list(start_sequence)
    for _ in range(num_notes):
        input_seq = np.expand_dims(generated[-len(start_sequence):], axis=0)
        predicted_note = model.predict(input_seq)
        generated.append(predicted_note)
    return np.array(generated)

def notes_to_midi(notes, output_file='generated_music.mid'):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for note in notes:
        start, end, pitch = note
        midi_note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=float(start), end=float(end))
        instrument.notes.append(midi_note)
    midi.instruments.append(instrument)
    midi.write(output_file)
