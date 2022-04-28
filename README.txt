README: Generative Learning for Microbiome Data

This directory includes example files for data cleaning, creation, and implementation of deep learning for microbiome data.
Code was written in python version 3.7.0, using Tensorflow and Keras version 2.

The directory contains five python files, three jupyter notebooks, and one explainer document. 

The python files are background code, defining the classes which are run in each jupyter notebook.
callbacks.py contains custom callback functions for the VAE
utils.py contains utility functions for the MB-GAN
MicrobeVAE.py defines a VAE class
MicrobeWGANGP.py defines a WGAN-GP class
MBGAN.py defines an MBGAN class


The jupyter notebooks contain annotated code for three classses of generative models: Variational Autoencoder, Generative Adversarial Network, and MBGAN.
fracking_treatment_split.ipynb contains example code for separating a dataset by treatment groups
FrackingVAE.ipynb contains example code for defining and training a VAE
FrackingWGANGP.ipynb contains example code for defining and training a WGANGP
mbgan_test.ipynb contains example code for defining and training an MBGAN

The explainer document briefly discusses the mathematics behind the these architectures, and it then explains their implementations in greater depth.

Code in these files was modified from the following github directories:
https://github.com/davidADSP/GDL_code
https://github.com/zhanxw/MB-GAN
