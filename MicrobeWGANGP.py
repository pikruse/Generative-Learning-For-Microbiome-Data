#!/usr/bin/env python
# coding: utf-8

# In[6]:


from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from tensorflow.keras.layers import Layer

from keras.models import Model, Sequential
from keras import backend as K
import keras.optimizers
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model
from keras.initializers import RandomNormal

from functools import partial

import tensorflow as tf
import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt

class RandomWeightedAverage(Layer):
    def __init__(self, batch_size):
        super().__init__() #init class without having to call _Merge class
        self.batch_size = batch_size
  
    # provides random weighted avg. between real and generated microbe samples
    def call(self, inputs):
        #take random, positive uniform sample of shape batch_size, 1, 1 (tensor)
        #this serves as our random 'distance' between the real and generated samples
        alpha = tf.random.uniform((self.batch_size, 1, 1, 1)) 
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1]) #multiply alpha and 1 - alpha with real and fake samples
    
    def compute_output_shape(self, input_shape):
        # take the computed interpolated sample and return its first dimension
        return input_shape[0]
    
class PhyloTransform(Layer):
    def __init__(self, tf_matrix=None, **kwargs):
        if tf_matrix is None:
            self.kernel = None
        else:
            self.output_dim = tf_matrix.shape[1:]
            self.kernel = K.constant(tf_matrix, dtype='float32')
        super(PhyloTransform, self).__init__(**kwargs)
        
    def call(self, x):
        if self.kernel is None:
            return x
        else:
            return K.dot(x, self.kernel)
        
    def compute_output_shape(self, input_shape):
        if self.kernel is None:
            return input_shape
        else:
            return (input_shape[0], ) + self.output_dim       

class WGANGP():
    def __init__(self, input_dim,
                 critic_dense_neurons,
                 critic_batch_norm_momentum,
                 critic_activation,
                 critic_dropout_rate,
                 critic_learning_rate,
                 generator_initial_dense_layer_size,
                 generator_dense_neurons,
                 generator_batch_norm_momentum,
                 generator_activation,
                 generator_dropout_rate,
                 generator_learning_rate,
                 optimizer,
                 grad_weight,
                 z_dim,
                 batch_size):
        
        # initiate attributes
        self.name = 'gan'
        
        self.input_dim = input_dim
        self.critic_dense_neurons = critic_dense_neurons
        self.critic_batch_norm_momentum = critic_batch_norm_momentum
        self.critic_activation = critic_activation
        self.critic_dropout_rate = critic_dropout_rate
        self.critic_learning_rate = critic_learning_rate
        
        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_dense_neurons = generator_dense_neurons
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate
        
        self.n_layers_critic = len(critic_dense_neurons)
        self.n_layers_generator = len(generator_dense_neurons)
        
        self.optimizer = optimizer
        
        self.z_dim = z_dim
        
        self.weight_init = RandomNormal(mean=0, stddev=0.02)
        self.grad_weight = grad_weight
        self.batch_size = batch_size
        
        #make list to contain losses
        self.d_losses = []
        self.g_losses = []
        self.epoch = 0
        
        self._build_critic()
        self._build_generator()
        
        self._build_adversarial()
    
    def gradient_penalty_loss(self, y_true, y_pred, interpolated_samples):
        
        # compute Gradient Penalty based on predictions + real/fake weighted samples
        gradients = K.gradients(y_pred, interpolated_samples)[0]
        
        #compute euclidian norm by squaring,
        gradients_sqr = K.square(gradients)
        # summing squares over the rows
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))               
        # and square rooting
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)   
        
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1-gradient_l2_norm)
        
        #return mean as loss over all batch samples
        return K.mean(gradient_penalty)
        
    def wasserstein(self, y_true, y_pred):
        # define wasserstein loss
        return K.mean(y_true * y_pred)
    
    def get_activation(self, activation):
        # instantiate activation function - if leaky relu, set alpha to 0.2, if not, leave default
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha=0.2)
        else:
            layer = Activation(activation)
        return layer
    
    def get_optimizer(optimizer, lr, decay=0.0, clipnorm=0.0, clipvalue=0.0, **kwargs):
        # get optimizer
        support_optimizers = {'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam'}
        assert optimizer in support_optimizers
        fn = getattr(keras.optimizers, optimizer)
        return fn(lr, decay=decay, clipnorm=clipnorm, clipvalue=clipvalue, **kwargs)
    
    def _build_critic(self):
        ## Build the Critic
        # input size should be the same as our sample size
        critic_input = Input(shape=self.input_dim, name='critic_input') 
        x = critic_input
        
        x = PhyloTransform(
        
        # for loop to create dense layers
        for i in range(self.n_layers_critic):
            x = Dense(self.critic_dense_neurons[i], name='critic_dense_'+str(i),
                      kernel_initializer = self.weight_init)(x)
            
            if self.critic_batch_norm_momentum and i > 0:
                x = BatchNormalization(self.critic_batch_norm_momentum)(x)
                
            x = self.get_activation(self.critic_activation)(x)
            
            if self.critic_dropout_rate:
                x = Dropout(rate=self.critic_dropout_rate)(x)
        
        # critic output is a binary classifier - is a sample REAL or FAKE?
        critic_output = Dense(1, activation=None, kernel_initializer = self.weight_init)(x)
        
        # instantiate model with critic input as input and critic output as output
        self.critic = Model(critic_input, critic_output)
    
    def _build_generator(self):
        # Build the generator
        generator_input = Input(shape=(self.z_dim,), name='generator_input')
        x = generator_input
        
        #create initial dense layer
        x = Dense(self.generator_initial_dense_layer_size, kernel_initializer=self.weight_init)(x)
        
        if self.generator_batch_norm_momentum:
            x = BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)
        
        x = self.get_activation(self.generator_activation)(x)
        
        if self.generator_dropout_rate:
            x = Dropout(rate = self.generator_dropout_rate)(x)
        
        # loop to create rest of layers
        for i in range(self.n_layers_generator):
            x = Dense(self.generator_dense_neurons[i], name="generator_dense_"+str(i))(x)
            
            if i < self.n_layers_generator - 1:
                if self.generator_batch_norm_momentum:
                    x = BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)
                x = self.get_activation(self.generator_activation)(x)
            else:
                x = Activation('tanh')(x)
            
        generator_output = x
        
        # instantiate model with generator input as input, gen. output as output
        self.generator = Model(generator_input, generator_output)
    
    def get_opti(self, lr):
        #instantiate opt
        if self.optimizer == 'adam':
            opti = Adam(lr=lr, beta_1 = 0.5)
        elif self.optimizer == 'rmsprop':
            opti = RMSprop(lr=lr)
        else:
            opti = Adam(lr=lr)
        
        return opti
    
    def set_trainable(self, m, val):
        # turn training on/off for generator
        m.trainable = val
        for l in m.layers:
            l.trainable = val
            
    def _build_adversarial(self):
        
        # construct computational graph for critic??
        
        #freeze generator while training critic
        self.set_trainable(self.generator, False)
        
        # real sample input
        real_smp = Input(shape=self.input_dim)
        
        # fake sample input from z dimension
        z_disc = Input(shape=(self.z_dim,))
        # call generator to create fake sample
        fake_smp = self.generator(z_disc)
        
        # critic determines validity of real and fake samples
        fake = self.critic(fake_smp)
        valid = self.critic(real_smp)
        
        # construct weighted avg. between real and fake samples
        # weighted average = interpolated sample, between real and fake
        interpolated_smp = RandomWeightedAverage(self.batch_size)([real_smp, fake_smp])
        
        # determine validity of interpolated sample
        validity_interpolated = self.critic(interpolated_smp)
        
        # w/ Python Partial, create loss fn with 'interpolated sample' argument
        # use our gradient penalty loss as the base and input interpolated samples so we can compute gradients from them
        partial_gp_loss = partial(self.gradient_penalty_loss, interpolated_samples = interpolated_smp)
        partial_gp_loss.__name__ = 'gradient_penalty' #keras requires fn names
        
        # initialize full critic model
        # inputs are real and fake sample, while outputs are likelihood that valid, fake, and interpolated samples are real
        self.critic_model = Model(inputs=[real_smp, z_disc],
                                  outputs=[valid, fake, validity_interpolated])
        
        # compile model with wasserstein and gp loss, and weight gp loss how we see fit
        self.critic_model.compile(loss=[self.wasserstein, self.wasserstein, partial_gp_loss],
                                  optimizer=self.get_opti(self.critic_learning_rate),
                                  loss_weights=[1, 1, self.grad_weight])
        
        # construct computational graph for generator
        
        #for generator, freeze critic layers
        self.set_trainable(self.critic, False)
        self.set_trainable(self.generator, True)
        
        # sampled noise for input to generator
        model_input = Input(shape=(self.z_dim,))
        # generate samples from noise
        smp = self.generator(model_input)
        #discriminator determines validity
        model_output = self.critic(smp)
        
        # define generator model
        self.model = Model(model_input, model_output)
        self.model.compile(optimizer = self.get_opti(self.generator_learning_rate),
                           loss=self.wasserstein)
        
        # turn critic back to trainable
        self.set_trainable(self.critic, True)
    
    def train_critic(self, x_train, batch_size, using_generator):
        
        # get response vectors for critic: valid samples are 1, fake samples are negative 1
        # instantiate dummy vector of ones as well for interpolated samples
        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = -np.ones((batch_size, 1), dtype=np.float32)
        dummy = np.ones((batch_size, 1), dtype=np.float32)
        
        # if using a data generator (not a dataset, but feeding samples one at a time from directory), use the next() function
        if using_generator:
            true_smps = next(x_train)[0]
            if true_smps.shape != batch_size:
                true_smps = next(x_train)[0]
                
        # if not using a generator, this code runs:
        else:
            # take a random sample of indexes of size batch sizefrom the training data
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            # select true tramples using this randomly generated index
            true_smps = x_train.iloc[idx,]
        
        # generate batch_size number of samples of gaussian noise with z_dim length
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        # train critic, inputting true samples and noise and outputting truth likelihood for valid, fake, and interpolated samples
        d_loss = self.critic_model.train_on_batch([true_smps, noise], [valid, fake, dummy])
        
        return d_loss
    
    def train_generator(self, batch_size):
        ## Train generator
        
        # create vector of valid outputs to associate 
        valid = np.ones((batch_size, 1), dtype=np.float32)
        # and a noise vector to create samples 
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)
    
    def train(self, x_train, batch_size, epochs, print_every_n_batches = 10,
                    n_critic = 5, using_generator = False):
        # overall training
        for epoch in range(self.epoch, self.epoch + epochs):
            if epoch % 100 == 0:
                critic_loops = 5
            else:
                critic_loops = n_critic
            
            # train critic first for the specified number of given in n_critic
            for _ in range(critic_loops):
                d_loss = self.train_critic(x_train, batch_size, using_generator)
            
            # then train generator for one step
            g_loss = self.train_generator(batch_size)
            
            # print epoch, # critic loops, D loss and breakdown, and generator loss
            print("%d, (%d, %d) [D Loss: (%.1f)(R %.1f, F %.1f, I %.1f)] [G loss: %.1f]" % 
                  (epoch, critic_loops, 1, d_loss[0], d_loss[1],d_loss[2],d_loss[3],g_loss))
            
            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)
            

