from socketserver import ThreadingMixIn
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import numpy as np

import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


st.write("# PDM and StreamLit Example")

st.write("In this example we have 4 different datasets.")
option = st.selectbox(
     'How would you like to be contacted?',
     ('FD001', 'FD002', 'FD003', 'FD004'))
st.write("You select:",option)

# Defining the coloumns
col = {0:'engineNumber',1:'cycleNumber',2:'opSetting1',3:'opSetting2',4:'opSetting3',5:'sensor1',6:'sensor2',
           7:'sensor3',8:'sensor4',9:'sensor5',10:'sensor6',11:'sensor7',12:'sensor8',13:'sensor9',14:'sensor10',
           15:'sensor11',16:'sensor12',17:'sensor13',18:'sensor14',19:'sensor15',20:'sensor16',
           21:'sensor17',22:'sensor18',23:'sensor19',24:'sensor20',25:'sensor21'}

# Loading the data
train_data  = pd.read_csv(f"./CMAPS/train_{option}.txt", sep = "\s+",  header=None)
rul_data    =  pd.read_csv(f"./CMAPS/rul_{option}.txt",  sep = "\s+",header = None)
test_data   = pd.read_csv(f"./CMAPS/test_{option}.txt",  sep = "\s+",header = None)

# Renaming the columns
train_data = train_data.rename(columns=col)
test_data = test_data.rename(columns=col)
rul_data = rul_data.rename(columns={0: 'RUL'})

# Extracting the different types of columns
sensor_columns = [col for col in train_data.columns if col.startswith("sensor")]
setting_columns = [col for col in train_data.columns if col.startswith("setting")]
st.write("Training data:")
st.write(train_data.head())

st.write("This is what the RUL data looks like:")
st.write(rul_data.head())

n_turb = train_data["engineNumber"].unique().max()

st.write(f"There are {n_turb} unique turbines in the dataset.")
# extract the first unit from the first dataset

rul = pd.DataFrame(train_data.groupby('engineNumber')['cycleNumber'].max()).reset_index()
rul.columns = ['engineNumber', 'max']
train_data = train_data.merge(rul, on=['engineNumber'], how='left')
train_data['RUL'] = train_data['max'] - train_data['cycleNumber']
train_data.drop('max', axis=1, inplace=True)
# The different weights
w1 = 30
w0 = 15
train_data['label1'] = np.where(train_data['RUL'] <= w1, 1, 0 )
train_data['label2'] = train_data['label1']
train_data.loc[train_data['RUL'] <= w0, 'label2'] = 2

# generate RUL
rul = pd.DataFrame(test_data.groupby('engineNumber')['cycleNumber'].max()).reset_index()
rul.columns = ['engineNumber', 'max']
rul_data.columns = ['more']
rul_data['engineNumber'] = rul_data.index + 1
rul_data['max'] = rul['max'] + rul_data['more']
rul_data.drop('more', axis=1, inplace=True)
test_data = test_data.merge(rul_data, on=['engineNumber'], how='left')
test_data['RUL'] = test_data['max'] - test_data['cycleNumber']
test_data.drop('max', axis=1, inplace=True)

# generate label columns w0 and w1 for test data
test_data['label1'] = np.where(test_data['RUL'] <= w1, 1, 0 )
test_data['label2'] = test_data['label1']
test_data.loc[test_data['RUL'] <= w0, 'label2'] = 2

# Ask user to pick a window
window = st.slider("Pick a window size of cycles", 1,101)

# function to reshape features into (samples, time steps, features) 

def reshapeFeatures(id_df, seq_length, seq_cols):
    # sourcery skip: remove-zero-from-range
    """
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length.
    An alternative would be to pad sequences so that
    we can use shorter ones.
    
    :param id_df: the data set to modify
    :param seq_length: the length of the window
    :param seq_cols: the columns concerned by the step
    :return: a generator of the sequences
    """
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]
        
        
# pick the feature columns 
# sourcery skip: merge-list-extend
sensor_cols = ['sensor' + str(i) for i in range(1,22)]
sequence_cols = ['opSetting1', 'opSetting2', 'opSetting3', 'cycleNumber']
sequence_cols.extend(sensor_cols)

# generator for the sequences
feat_gen = (list(reshapeFeatures(train_data[train_data['engineNumber']==engineNumber], window, sequence_cols)) 
           for engineNumber in range(1, n_turb + 1))

# generate sequences and convert to numpy array
feat_array = np.concatenate(list(feat_gen)).astype(np.float32)


# function to generate label

def reshapeLabel(id_df, seq_length=window, label=['RUL']):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length."""
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length: num_elements, :]

# generate labels
label_gen = [reshapeLabel(train_data[train_data['engineNumber']==engineNumber]) for engineNumber in range(1, n_turb + 1)]

label_array = np.concatenate(label_gen).astype(np.float32)


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

nb_features = feat_array.shape[2]
nb_out = label_array.shape[1]

# Forbereder data for testing
# We pick the last sequence for each engineNumber in the test data
seq_array_test_last = [test_data[test_data['engineNumber']==engineNumber][sequence_cols].values[-window:] 
                       for engineNumber in range(1, n_turb + 1) if len(test_data[test_data['engineNumber']==engineNumber]) >= window]
seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
y_mask = [len(test_data[test_data['engineNumber']==engineNumber]) >= window for engineNumber in test_data['engineNumber'].unique()]
label_array_test_last = test_data.groupby('engineNumber')['RUL'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)


# Funksjon for å lage en model


def create_model():
    model = Sequential()
    model.add(LSTM(input_shape=(window, nb_features), units=100, return_sequences=True, name=f"lstm_0"))
    model.add(Dropout(0.2, name=f"dropout_0"))
    model.add(LSTM(units=50, return_sequences=True, name="lstm_1"))
    model.add(Dropout(0.2, name="dropout_1"))
    model.add(LSTM(units=25, return_sequences=False, name="lstm_2"))
    model.add(Dropout(0.2, name="dropout_2"))
    model.add(Dense(units=nb_out, name="dense_0"))
    model.add(Activation("linear", name="activation_0"))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[root_mean_squared_error, 'mae'])
    summary = []
    model.summary(print_fn = lambda x: summary.append(x))
    summary = "\n".join(summary)
    return model

# Definerer variabler som er tilgjengelig i hele skriptet
global epochs
global batch_size
epochs = st.slider("Pick an epoch size", 1,100)
batch_size = st.slider("Pick an batch size", 1,100)


def train_model(model):
    #epochs = 100
    #batch_size = 200

    # fit the network
    history = model.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, validation_split=0.05, verbose=1,
              callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                         verbose=0, mode='min'),
                           keras.callbacks.ModelCheckpoint("model.h5", monitor='val_loss',
                                                           save_best_only=True, mode='min', verbose=0)]
              )
    return history

global model
global hist


def create_buttons():
    global model
    if st.button("Create , Train, and test the Model"):
        st.write("Creating model...")
        model = create_model()
        st.write("Model Created")
        global hist 
        st.write("Training model...")
        hist = train_model(model)
        st.write("Training is done!")
        st.write("Printing history...")
        st.write( hist.history.keys() )
        st.write("The Root mean Squared Error error for each Epoch is:" 
        ,hist.history['root_mean_squared_error'])
        st.write("Testing the model now...")
        y_pred_test = model.predict(seq_array_test_last)
        y_true_test = label_array_test_last
        scores_test = model.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
        s1 = ((y_pred_test - y_true_test)**2).sum()
        moy = y_pred_test.mean()
        s2 = ((y_pred_test - moy)**2).sum()
        s = 1 - s1/s2
        st.write('\nEfficiency: {}%'.format(s * 100))
    

    

