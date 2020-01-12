#!/home/xqding/apps/miniconda3/bin/python

import numpy
import torch
from VAE_model import *
import pickle
import argparse
from sys import exit

## read training to get the length of protein and the number of
## amino acide types
with open("./output/seq_msa_binary.pkl", 'rb') as file_handle:
    seq_msa_binary = pickle.load(file_handle)
len_protein = seq_msa_binary.shape[1]
num_aa_type = seq_msa_binary.shape[2]

## load trained VAE model    
vae = VAE(21, 2, len_protein * num_aa_type, [100])
model_state_dict = torch.load("./output/model/vae_0.01.model")
vae.load_state_dict(model_state_dict)

## define points in latent space to be converted into sequences
## here I will just use random points as an example
num_seqs = 5
dim_latent_space = 2
z = torch.randn(num_seqs, dim_latent_space)
with torch.no_grad():
    log_p = vae.decoder(z)
    p = torch.exp(log_p)
    p = torch.reshape(p, (num_seqs, len_protein, num_aa_type))
p = p.numpy()

## p is a numpy array with shape [num_seqs, len_protein, num_aa_type].
## p[i,:,:] represents the sequence converted from the latent space point z[i,:].
## p[i,:,:] represents the probabilities of each amino acid at all positions of the i'th sequence.
## Therefore, each point in the latent space is converted into a distribution of protein sequences.
## This distribution is defined by probabilities of amino acid types at all positions.
## p[i,j,:] is a vector of 21 probablities represent the probablity of 21 amino acid type
## at the j'th position of the i'th sequence, so np.sum(p[i,j,:]) = 1
assert(np.all(np.sum(p, -1) - 1 <= 1e-6))

## the mapping between positions and amino acid types are stored in aa_index
with open("./output/aa_index.pkl", 'rb') as file_handle:
    aa_index = pickle.load(file_handle)
idx_to_aa_dict = {}
idx_to_aa_list = ['' for i in range(num_aa_type)]
for k, v in aa_index.items():
    idx_to_aa_dict[v] = k
    idx_to_aa_list[v] = k

## we can convert probablities into actual sequences by choosing the most likely sequences.
max_prob_idx = np.argmax(p, -1)
seqs = []
for i in range(num_seqs):
    seq = [idx_to_aa_list[idx] for idx in max_prob_idx[i,:]]
    seqs.append("".join(seq))

## the converted most likely sequences are in seqs
for i in range(num_seqs):
    print(seqs[i])

## we can also get the second, third most likely sequences from p for each point in the latent space.

