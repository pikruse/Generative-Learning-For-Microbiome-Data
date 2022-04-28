#!/usr/bin/env python
# coding: utf-8

# import necessary packages
from keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import os

# creat custom callback class
class CustomCallback(Callback): # custom callback class built on top of keras's callback class
    
    # define attributes for class
    def __init__(self, run_folder, print_every_n_batches, initial_epoch, vae):
        self.epoch = initial_epoch
        self.run_folder = run_folder
        self.print_every_n_batches = print_every_n_batches
        self.vae = vae
    
    # define model behavior on batch end
    def on_batch_end(self, batch, logs={}):
        if batch % self.print_every_n_batches == 0:
            z_new = np.random.normal(size = (1, self.vae.z_dim)) #randomly sample a vector of length z_dim from a normal dist.
            
            # make predictions with this sample using the decoder
            # turn random sample into an array, remove axes of dim. 1, and make a prediction
            reconst = self.vae.decoder.predict(np.array(z_new))[0].squeeze()
    
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch += 1
        
def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    # wrapper function to create LearningRateScheduler with step decay schedule
    
    def schedule(epoch):
        # decay function for lr
        new_lr = initial_lr * (decay_factor ** np.floor(epoch/step_size))
        return new_lr
    
    # return LearningRateScheduler with step decay built in
    return LearningRateScheduler(schedule)

