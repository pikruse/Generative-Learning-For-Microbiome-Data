#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import necessary packages
from tensorflow.keras.layers import Input, Conv1D, Conv2D, Flatten, Dense, Conv1DTranspose, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

# import functions defined in callbacks.py script
from callbacks import CustomCallback, step_decay_schedule

import numpy as np
import json
import os 
import pickle 

# define variational autoencoder class
class VariationalAutoencoder():
    
    #attributes
    def __init__(self,
                 input_dim,
                 encoder_layer_size,
                 decoder_layer_size,
                 z_dim,
                 use_batch_norm = False,
                 use_dropout = False):
        self.name = 'variational_autoencoder'
        
        self.input_dim = input_dim
        self.encoder_layer_size = encoder_layer_size
        self.decoder_layer_size = decoder_layer_size
        self.z_dim = z_dim
        
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        
        # number of layers in VAE is equivalent to length of list of conv filters
        self.n_layers_encoder = len(encoder_layer_size)
        self.n_layers_decoder = len(decoder_layer_size)
        
        self._build()
        
    # build the VAE
    def _build(self):
            
        # ENCODER
        encoder_input = Input(shape = self.input_dim, name = 'encoder_input')
            
        x = encoder_input
        
        # create the layers in the encoder using a for loop
        for i in range(self.n_layers_encoder):
                
            # use dense for vector input
            enc_dense_layer = Dense(self.encoder_layer_size[i],
                                    name = 'encoder_dense_' + str(i))
            
            x = enc_dense_layer(x)
                
            # add batch norm (if true)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
                
            # add activation
            x = LeakyReLU()(x)
                
            # add dropout (if true)
            if self.use_dropout:
                x = Dropout(0.25)(x)
            
        # encoder has two outputs, mu and log_var, which are used to make a distribution
        self.mu = Dense(self.z_dim, name = 'mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)
            
        # define the model w/ encoder inputs as input and mu + log_var as output
        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))
            
        # create sampling function, which samples from the distribution defined by mu and log_var
        def sampling(args):
            mu, log_var = args
            # sample of shape mu from the std. normal to give variance to distribution
            epsilon = K.random_normal(shape = K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon
                
        # turn sampling function into keras layer
        encoder_output = Lambda(sampling, name = 'encoder_output')([self.mu, self.log_var])
            
        # define full model
        self.encoder = Model(encoder_input, encoder_output)
            
            
        # DECODER
            
        # receive output from encoder as input to decoder
        decoder_input = Input(shape = (self.z_dim,), name = 'decoder_input')
            
        x = decoder_input
    
        # loop to create decoder layers
        for i in range(self.n_layers_decoder):
            dec_dense_layer = Dense(self.decoder_layer_size[i],
                                    name = 'decoder_dense_' + str(i))
                
            x = dec_dense_layer(x)
                
            # if we are not on the last layer, use batch norm / dropout if true and add activation
            # if last layer, use ReLU activation (since sample values cannot be 0)
            if i < self.n_layers_decoder - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                    
                if self.use_dropout:
                    x = Dropout(rate = 0.25)(x)
            else:
                x = Activation('relu')(x)
                                   
        decoder_output = x    
        self.decoder = Model(decoder_input, decoder_output)
            
            
        # FULL VAE
            
        # Model takes in encoder input as input and outputs the decoded encoder input
        model_input = encoder_input
        model_output = self.decoder(encoder_output)
            
        self.model = Model(model_input, model_output)
            
    #compile function
    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate
            
        #reconstruction loss
        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred))
            return r_loss_factor * r_loss
            
        # kullback-liebler divergence
        def vae_kl_loss(y_true, y_pred):
            kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
            return kl_loss
            
        # total loss
        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return r_loss + kl_loss
            
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = vae_loss, metrics = [vae_r_loss, vae_kl_loss])
        
    # training function
    def train(self, x_train, y_train, batch_size, epochs, lr_decay = 1):
            
        # learning rate schedule
        lr_sched = step_decay_schedule(initial_lr = self.learning_rate, decay_factor = lr_decay, step_size = 1)
        callbacks = [lr_sched]
            
        self.model.fit(x_train, y_train,
                       epochs = epochs,
                       callbacks = callbacks,
                       batch_size = batch_size, 
                       validation_split = .16)
            
            

