"""In this file, we use our trained Neural Network
to generate music using what it's learned from our
input sequences of note and chord objects"""

import numpy as np
import pickle
from music21 import instrument, note, stream, chord
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


'''The following function with generate a MIDI file'''
def generate():

    #load the note objects training our model using pickle
    with open('data/notes', 'rb') as filepath:
        notes_array = pickle.load(filepath)

    #get the pitch names(A-G)
    pitchnames = sorted(set(item for item in notes_array))
    n_names = len(set(notes_array))

    # define norm_input and net_input for LSTM Neural Network
    # and call helper function to create the model, the output,
    # and generate the MIDI file
    net_input, norm_input = prep_sequences(notes_array, pitchnames,   n_names)
    model = create_model(net_input, n_names)
    predict_output = gen_notes(model, net_input, pitchnames, n_names)
    gen_midi(predict_output)

'''Here we prepare the sequences used by our NN'''
def prep_sequences(notes_array, pitchnames, n_names):

    # dictionary as storage for mapping the pitches(A-G
    pitch_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # define our length of sequences and our other variables
    sequence_len = 100
    net_input = []
    out = []


    for i in range(0, len(notes_array) - sequence_len, 1):
        sequence_in = notes_array[i:i + sequence_len]
        sequence_out = notes_array[i + sequence_len]
        net_input.append([pitch_to_int[char] for char in sequence_in])
        out.append(pitch_to_int[sequence_out])

    note_patterns = len(net_input)

    norm_input = np.reshape(net_input, (note_patterns, sequence_len, 1))
    norm_input = norm_input / float(n_names)

    return (net_input, norm_input)

