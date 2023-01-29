"""
Copyright (C) 2023 Kristiania University College- All Rights Reserved
You may use, distribute and modify this code under the
terms of the MIT license.
You should have received a copy of the MIT license with
this file. If not, please write to: https://opensource.org/licenses/MIT

Project: ENViSEC - Artificial Intelligence-enabled Cybersecurity for Future Smart Environments 
(funded from the European Union’s Horizon 2020, NGI-POINTER under grant agreement No 871528).
@Authors: Guru Bhandari, Andreas Lyth, Andrii Shalaginov, Tor-Morten Grønli
@Programmer: Guru Bhandari
@File - definition of different ML models.
"""

import tensorflow as tf
from tensorflow.keras.layers import (Dense,
                                     Dropout)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding, Flatten, BatchNormalization

# # Creating the tf model
# def model_dnn(input_dim, output_dim):
#     model = Sequential()
#     model.add(Dense(12, input_dim = input_dim, activation = 'relu'))
#     model.add(Dense(8, activation = 'relu'))
#     # model.add(Dense(8, activation = 'relu'))
#     model.add(Dense(output_dim, activation = 'sigmoid'))
#     return model

def create_DNN(input_dim, output_dim):
    # create model
    model = Sequential()
    model.add(Dense(
        units=64, 
        input_dim=input_dim,
        kernel_initializer='he_normal', 
        activation='relu')
        )
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu')),
    # model.add(BatchNormalization())
    # model.add(Dropout(0.25))
    # model.add(Dense(32,   kernel_initializer="glorot_normal", activation='relu')),
    # model.add(Dropout(0.1))
    model.add(Dense(output_dim, activation='softmax'))
    return model


# Creating the tf model for PoC Current Sensing
def model_dnn_led():
    model = Sequential([
        # using Flatten isn't necessary, but included as it's handy for building out.
        Flatten(input_shape=(13,)),
        # started at 128, but we don't need that many -MC
        Dense(64, activation=tf.nn.relu),
        # Second layer for betterment of humanity... -MC
        Dense(64, activation=tf.nn.relu),
        Dense(1, activation=tf.nn.sigmoid)
    ])
    return model


def create_multiDNN(input_dim, output_size, dropout_rate):
    """
    multi-layer DNN model for the training
    """
    model = Sequential()
    model.add(Dense(2000, activation='relu',input_dim=input_dim))
    model.add(Dense(1500, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(800,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(400,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(150,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_size, activation='softmax'))
    print(model.summary())
    return model


def create_LSTM(input_dim, output_size, dropout_rate):
    """
    multi-layer DNN model for the training
    """
    model = Sequential()
    # First LSTM layer defining the input sequence length
    model.add(LSTM(input_shape=(30-1, 1),
                   units=32,
                   return_sequences=True))
    model.add(Dropout(0.2))

    # Second LSTM layer with 128 units
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.2))

    # Third LSTM layer with 100 units
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_size, activation='softmax'))

    print(model.summary())
    return model




