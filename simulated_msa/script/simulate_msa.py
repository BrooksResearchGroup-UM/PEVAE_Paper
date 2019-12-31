__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/02/10 01:18:33"

import numpy as np
import numpy.random as nrand
from ete3 import Tree
import pickle
import random
from sys import exit

## set the random seed so that the simulate is reproducible
## you can delete the following two lines of command if
## you want to use different random seeds in different runs
random.seed(0)
nrand.seed(0)

## read the random phylogentic tree
t = Tree("./output/random_tree.newick", format = 1)
t.name = str(len(t))

## read LG replacement matrix
with open("./output/LG_matrix.pkl", 'rb') as file_handle:
    LG_matrix = pickle.load(file_handle)
R_dict = LG_matrix['R_dict']
Q_dict = LG_matrix['Q_dict']
PI_dict = LG_matrix['PI_dict']
R = LG_matrix['R']
Q = LG_matrix['Q']
PI = LG_matrix['PI']
amino_acids = LG_matrix['amino_acids']
aa2idx = {}
for i in range(len(amino_acids)):
    aa2idx[amino_acids[i]] = i
    
## sample sequence for the root node from the equilibrium
## distribution of amino acids
len_protein = 100
root_seq = nrand.choice(amino_acids,
                        size = len_protein,
                        replace = True,
                        p = PI.reshape(-1) / np.sum(PI))
t.add_feature('seq', root_seq)

## simulate sequences for each node
## the evolution process is modelled as a continous-time Markov chain.
## the following script is used for simulating the continous-time Markov chain.
for node in t.traverse('preorder'):    
    if node.is_root():
        continue    
    anc_node = node.up
    
    seq = np.copy(anc_node.seq)
    dist = node.dist

    while True:
        tot_rate = -np.sum([Q_dict[(aa, aa)] for aa in seq])                
        wait_time = nrand.exponential(scale = 1/tot_rate)
        
        if wait_time > dist: break
        
        idx_prob = np.array([-Q_dict[(aa, aa)] for aa in seq]) / tot_rate        
        idx = nrand.choice(range(len_protein), p = idx_prob)

        aa_idx = aa2idx[seq[idx]]
        aa_type_prob = Q[aa_idx,:] / (-Q[aa_idx, aa_idx])
        aa_type_prob[aa_idx] = 0
        
        aa_mutant = nrand.choice(amino_acids, p = aa_type_prob)
        seq[idx] = aa_mutant
        dist -= wait_time        
    node.add_feature('seq', seq)    

#### save the simulated sequences
simulated_msa = {}

## save the simulated sequences in a text file
with open("./output/simulated_msa.txt", 'w') as file_handle:
    for node in t.traverse('preorder'):
        print("> {:<10}".format(node.name), end = "", file = file_handle)
        seq = "".join(node.seq)
        print(seq, end = "", file = file_handle)
        print("", file = file_handle)
        simulated_msa[node.name] = seq

## save the simulated sequences in a pickle file
with open("./output/simulated_msa.pkl", 'wb') as file_handle:
    pickle.dump(simulated_msa, file = file_handle)
