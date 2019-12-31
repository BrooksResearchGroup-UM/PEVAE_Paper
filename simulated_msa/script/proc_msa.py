__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/05/10 19:51:12"

'''
Process the simulated multiple sequence alignment into required format that is 
used for training latent space models using VAE latet.
'''

import pickle
import sys
import numpy as np
from ete3 import Tree
from sys import exit

## read the simulated multiple sequence alignment
with open("./output/simulated_msa.pkl", 'rb') as file_handle:
    msa = pickle.load(file_handle)

## read the LG replacement matrix
with open("./output/LG_matrix.pkl", 'rb') as file_handle:
    LG_matrix = pickle.load(file_handle)
amino_acids = LG_matrix['amino_acids']
aa2idx = {}
for i in range(len(amino_acids)):
    aa2idx[amino_acids[i]] = i

## convert msa into numbers from 0 to 19
len_protein = len(msa[list(msa.keys())[0]])
num_seq = len(msa.keys())

msa_keys = list(msa.keys())
msa_keys.sort()
msa_nume = np.zeros((num_seq, len_protein), dtype = np.int)
for i in range(num_seq):
    msa_nume[i,:] = [aa2idx[s] for s in msa[msa_keys[i]]]

## convert msa into binary representation, i.e., using 0 or 1    
msa_binary = np.zeros((num_seq, len_protein, 20))
D = np.identity(20)
for i in range(num_seq):
    msa_binary[i,:,:] = D[msa_nume[i]]

## save processed msa
with open("./output/msa_nume.pkl", 'wb') as file_handle:
    pickle.dump(msa_nume, file = file_handle)

with open("./output/msa_binary.pkl", 'wb') as file_handle:
    pickle.dump(msa_binary, file = file_handle)

with open("./output/msa_keys.pkl", 'wb') as file_handle:
    pickle.dump(msa_keys, file = file_handle)

#### the simulated sequences are splitted into two classes:
#### leaf node sequences and non-leaf node sequences.
#### only leaf node sequences are used for training latent
#### space models.

## save leaf node sequences
t = Tree("./output/random_tree.newick", format = 1)
t.name = str(len(t))
num_leaf = len(t)
msa_leaf_keys = [str(i) for i in range(num_leaf)]
msa_leaf_keys.sort()

msa_leaf_nume = np.zeros((num_leaf, len_protein), dtype = np.int)
for i in range(num_leaf):
    msa_leaf_nume[i,:] = [aa2idx[s] for s in msa[msa_leaf_keys[i]]]

msa_leaf_binary = np.zeros((num_leaf, len_protein, 20))
D = np.identity(20)
for i in range(num_leaf):
    msa_leaf_binary[i,:,:] = D[msa_leaf_nume[i]]

with open("./output/msa_leaf_nume.pkl", 'wb') as file_handle:
    pickle.dump(msa_leaf_nume, file = file_handle)

with open("./output/msa_leaf_binary.pkl", 'wb') as file_handle:
    pickle.dump(msa_leaf_binary, file = file_handle)

with open("./output/msa_leaf_keys.pkl", 'wb') as file_handle:
    pickle.dump(msa_leaf_keys, file = file_handle)
