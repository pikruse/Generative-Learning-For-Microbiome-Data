import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from keras.utils import Sequence
from scipy.stats import describe

TOL = 1e-4

# define shannon entropy calculations for post-hoc evaluation
def shannon_entropy(x, tol=0.):
    return -np.sum(np.where(x > tol, x * np.log(x), 0), axis = -1)

# define sparsity function for post-hoc evaluation
def get_sparsity(x, tol = TOL):
    return np.sum(x <= tol, axis=-1)/x.shape[1]

# define a loading function for data in pickle (.pkl) format
def load_sample_pickle_data(filename="./data/raw_data.pkl"):
    ## Load raw dataset
    raw_data = pickle.load(open(filename, 'rb'))
    
    #normalize to taxa proportion
    dataset = raw_data.iloc[:,1:].values/100.
    
    # extract whether a sample is a 'case' or 'ctrl' sample
    labels = raw_data["group"].values
    
    #extract taxa names
    taxa_list = raw_data.columns[1:]
    
    # split data into treated and untreated groups
    data_o_case = dataset[labels == 'case']
    data_o_ctrl = dataset[labels == 'ctrl']
    
    return data_o_case, data_o_ctrl, taxa_list

# define function to load data from CSV
def load_data_from_csv(csv_file, seq_dep=100, **kwargs):
    ## read raw data
    data = pd.read_csv(csv_file, **kwargs)
    
    # extract only species info from csv
    names = [_ for _ in data.index if _.split(';')[-1].startswith('s__')]
    data_s = data.loc[names, :].transpose()
    
    #print data shape and shannon entropy
    print(data_s.shape)
    print(describe(shannon_entropy(data_s.values/seq_dep)))
    
    return data_s.values/seq_dep, data_s.columns

def expand_phylo(taxa_list):
    """ expand taxa to higher order. 
        adj_matrix, taxa_indices = expand_phylo(colnames)
    """
    # define blank dictionary, number of taxa, and blank list
    memo, Ntaxa, adj, = {}, len(taxa_list), []
    
    # loop through taxa list
    for i, taxa in enumerate(taxa_list):
        # in memo dictionary, give each taxa an index
        memo[taxa] = i
        
        # split taxa along '|' in taxa name
        trunc = taxa.split('|')
        
        # loop through each part of the taxa name
        for j in range(len(trunc)):
            
            # rejoin each taxa name section to the next
            p = '|'.join(trunc[:j+1])
            
            # if this taxa name doesn't exist yet, add it to our taxa dictionary
            if p not in memo:
                memo[p] = Ntaxa
                Ntaxa += 1
            
            # add taxa number and name to adj list
            adj.append((memo[taxa], memo[p]))
            
    # return adj, memo
    return adj, dict((v, k) for k, v in memo.items())

# turn our existing adj list into a dense matrix
def adjmatrix_to_dense(x, shape, val=1):
    # specify shape of matrix with 0 mask
    mask = np.zeros(shape)
    
    # turn x input into array
    x = np.array(x).transpose()
    
    # any time that a value appears in our matrix, make it a 1
    mask[tuple(x)] = val
    
    # return matrix
    return mask
    