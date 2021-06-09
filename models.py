import os 

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Lambda, Conv1D, Conv2D, LeakyReLU,ReLU, Dropout, Flatten, Dense, Activation, Reshape, Conv2DTranspose, Input


def custom_activation(output):
    logexpsum = K.sum(K.exp(output), axis=-1, keepdims=True)
    result= logexpsum/ (logexpsum+ 1.0)

    return result

def define_discriminator(n_classes, opt, in_shape=(1, 120, 1)):
    inp = Input(shape= in_shape)
    #li = Reshape((1, 120, 1))(inp)
    fe = Conv2D(filters=32, kernel_size=(1,5))(inp)
    fe = LeakyReLU()(fe)
    #fe = ReLU()(fe)
    fe = Conv2D(filters=32, kernel_size=(1,5))(fe)
    fe = LeakyReLU()(fe)
    #fe = ReLU()(fe)
    fe = Conv2D(filters=32, kernel_size=(1,5))(fe)
    fe = LeakyReLU()(fe)
    #fe = ReLU()(fe)
    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)
    fe = Dense(n_classes)(fe)
    #Classifer C
    c_out_layer = Activation('softmax')(fe)
    c_model = Model(inp, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #Discriminator D
    d_out_layer = Lambda(custom_activation)(fe)
    d_model = Model(inp, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return d_model, c_model 

def define_generator(latent_dim = 100):
    in_lat = Input(shape=(latent_dim,))
    n_nodes = 32 * 1 * 108
    gen = Dense(n_nodes)(in_lat)
    gen = ReLU()(gen)
    gen = Reshape((1, 108, 32))(gen)
    gen = Conv2DTranspose(filters = 32, kernel_size=(1,5), strides=1)(gen)
    gen = ReLU()(gen)   
    gen = Conv2DTranspose(filters = 32, kernel_size=(1,5), strides=1)(gen)
    gen = ReLU()(gen)
    gen = Conv2DTranspose(filters = 32, kernel_size=(1,5), strides=1)(gen)
    gen = ReLU()(gen)
    out_layer = Conv2DTranspose(filters = 1, kernel_size= (1, 5), strides=1, activation='tanh', padding='same')(gen)
    g_model = Model(in_lat, out_layer)

    return g_model

def define_GAN(g_model, d_model, opt):
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)    
    model.compile(loss='binary_crossentropy', optimizer=opt)
    #model.summary()
    return model
   
def CNN(n_classes, opt):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(1,5), input_shape=(1, 120, 1), activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(1,5), activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(1,5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return model


