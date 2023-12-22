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

''' Here we create our NN and shape the layers using Keras 
 and load the weights saved during the training phase into
 the model'''
def create_model(net_input, n_names):

    model = Sequential()

    # first layer
    model.add(LSTM(
        256, input_shape=(net_input.shape[1], net_input.shape[2]), 
        return_sequences= True))
    # second layer
    model.add(Dropout(0.3))
    # third layer(second LSTM layer)
    model.add(LSTM(256, return_sequences= True))
    # fourth layer
    model.add(Dropout(0.3))
    # fifth layer
    model.add(LSTM(256))
    # sixth layer
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    # seventh layer
    model.add(Dense(256))
    # eighth layer
    model.add(Activation('relu'))
    # ninth layer
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    # tenth layer
    model.add(Dense(n_names))
    # eleventh layer
    model.add(Activation('softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.load_weights('weights-from-training.hdf5')



'''Here we have a helper function that generates the notes
(note objects) which will be used by our model'''

def gen_notes(model, net_input, pitchnames, n_names):

    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(net_input)-1)

    # define the sequence length and the note pattern
    note_pattern = net_input[start]

    # generate notes
    for note_index in range(500):

        # reshape the input to be used by the NN
        note_pattern = np.reshape(note_pattern, (1, len(note_pattern), 1))
        note_pattern = note_pattern / float(n_names)

        # predict the next note
        prediction = model.predict(note_pattern, verbose=0)

        # get the index of the note with the highest probability
        index = np.argmax(prediction)

        # get the note with the highest probability
        result = pitchnames[index]

        # add the note to the output
        note_pattern.append(index)

    return note_pattern

'''below we convert prtediction output to note objects
 then create a MIDI file'''
def gen_midi(predict_output):

    offset = 0
    notes_output = []

    # analyze the predicted output then form note objects
    for scheme in predict_output:

        #if the scheme from the output is a chord:
        if ('.' in scheme) or scheme.isdigit():
            note_from_chord = scheme.split('.')
            notes_array = []

            for curr_note in note_from_chord:
                new_note = note.Note(int(curr_note))
                new_note.storedInstrument = instrument.Piano()
                notes_array.append(new_note)
            
            new_chord = chord.Chord(notes_array)
            new_chord.offset = offset
            notes_output.append(new_chord)
        
        #if the scheme from the output is a note:
        else:
            new_note = note.Note(scheme)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            notes_output.append(new_note)

        #increase offset to avoid stacking notes
        offset += 0.5
    
    midi_stream = stream.Stream(notes_output)
    midi_stream.write('midi', fp='output.mid')

#call the generate function
if __name__ == '__main__':
    generate()  