"""Below is a module that parses and contains MIDI file data
that will be used to train the LSTM Neural Network"""
import glob
import numpy as np
from music21 import converter, instrument, note, chord
from keras.utils import np_utils
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.callbacks import ModelCheckpoint


def train_model():

    notes_array = get_notes()  #here is our problem
    notes_array = get_notes()

    if not notes_array:
        print("No notes parsed")
        return

    n_names = len(set(notes_array))

    n_names = len(set(notes_array))
    net_input, net_output = prep_sequences(notes_array, n_names)
    model = create_model(net_input, n_names)
    train(model, net_input, net_output)

def get_notes():
    ''' parse through MIDI files and store all the data
    (note objects and chord objects) into an array
    for training our LSTM Neural Network '''

    notes_array = []

    for file in glob.glob("midi_files/*.mid"):

        # get list of all note and chord objects in the file
        midi = converter.parse(file)

        print("Currently parsing %s" % file)
        parsed_notes = None

        components = instrument.partitionByInstrument(midi)

        if components: #midi file contains instruments
            parsed_notes = components.parts[0].recurse()
        else:
            parsed_notes = midi.flat.notes
        
        # for each component in the parsed data,
        # if a note or chord exists, we append
        # where each pitch of the note is appended as a string and each
        # chord is appended with the ID of every note (separated by a '.')
        # as a string
        for element in parsed_notes:
            if isinstance(element, note.Note):
                notes_array.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes_array.append('.'.join(str(n) for n in element.normalOrder))



""" Below we convert the categorical data stored in get_notes
into numerical data using a mapping function """

def prep_sequences(notes_array, n_names):
    ''' Prepare the sequences of note and chord objects into
    inputs that can be used to train our LSTM Neural Net'''

    sequence_len = 100

    # retrieve pitch names (A-G)
    pitchnames = sorted(set(i for i in notes_array))

    # dictionary as storage for mapping the pitches
    pitch_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    net_input = []
    net_output = []

    # form input and output sequences of NN training
    for i in range(0, len(notes_array) - sequence_len, 1):
        sequence_in = notes_array[i:i + sequence_len]
        sequence_out = notes_array[i + sequence_len]
        net_input.append([pitch_to_int[char] for char in sequence_in])
        net_output.append(pitch_to_int[sequence_out])

    note_patterns = len(net_input)

    # adjust input format for LSTM Neural Net layers
    net_input = np.reshape(net_input, (note_patterns, sequence_len, 1))
    net_input = net_input / float(n_names)
    net_output = np_utils.to_categorical(net_output)

    #return input and output sequences
    return (net_input, net_output)


"""Below we are going to develop our model, which contains
the LSTM layers, dropout layers, connected layers and the 
Activation later where the activation function for the NN
is found"""

def create_model(net_input, n_names):

    model = Sequential()

    # first layer
    model.add(LSTM(
        256, input_shape=(net_input.shape[1], net_input.shape[2]), 
        return_sequences= True))
    # second layer
    model.add(Dropout(0.3))
    # third layer(second LSTM layer)
    model.add(LSTM(256, return_sequences=True))
    # fourth layer(second dropout layer)
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_names))
    # activation layer
    model.add(Activation('softmax'))


    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


"""Below we proceed with training our model
and out input and output sequences formed above"""

def train(model, net_input, net_output):

    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    #checkpoint saves the weights of the nodes to a file(above)
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    #train the NN with 200 iterations and a batch size of 64
    model.fit(net_input, net_output, epochs=200, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    train_model()