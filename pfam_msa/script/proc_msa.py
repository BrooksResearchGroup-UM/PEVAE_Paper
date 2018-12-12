__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/05/10 19:51:12"

import pickle
import sys
import numpy as np
from sys import exit

file_name = "./MSA/PF00041_full.txt"
query_seq_id = "TENA_HUMAN/804-884"

## read all the sequences into a dictionary
seq_dict = {}
with open(file_name, 'r') as file_handle:
    for line in file_handle:
        if line[0] == "#" or line[0] == "/" or line[0] == "":
            continue
        line = line.strip()
        if len(line) == 0:
            continue
        seq_id, seq = line.split()
        seq_dict[seq_id] = seq.upper()

## remove gaps in the query sequences
query_seq = seq_dict[query_seq_id] ## with gaps
idx = [ s == "-" or s == "." for s in query_seq]
for k in seq_dict.keys():
    seq_dict[k] = [seq_dict[k][i] for i in range(len(seq_dict[k])) if idx[i] == False]
query_seq = seq_dict[query_seq_id] ## without gaps

## remove sequences with too many gaps
len_query_seq = len(query_seq)
seq_id = list(seq_dict.keys())
num_gaps = []
for k in seq_id:
    num_gaps.append(seq_dict[k].count("-") + seq_dict[k].count("."))
    if seq_dict[k].count("-") + seq_dict[k].count(".") > 10:
        seq_dict.pop(k)
        
with open("./output/seq_dict.pkl", 'wb') as file_handle:
    pickle.dump(seq_dict, file_handle)
        
## convert aa type into num 0-20
aa = ['R', 'H', 'K',
      'D', 'E',
      'S', 'T', 'N', 'Q',
      'C', 'G', 'P',
      'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
aa_index = {}
aa_index['-'] = 0
aa_index['.'] = 0
i = 1
for a in aa:
    aa_index[a] = i
    i += 1
with open("./output/" + "/aa_index.pkl", 'wb') as file_handle:
    pickle.dump(aa_index, file_handle)
    
seq_msa = []
keys_list = []
for k in seq_dict.keys():
    if seq_dict[k].count('X') > 0 or seq_dict[k].count('Z') > 0:
        continue    
    seq_msa.append([aa_index[s] for s in seq_dict[k]])
    keys_list.append(k)    
seq_msa = np.array(seq_msa)

with open("./output/keys_list.pkl", 'wb') as file_handle:
    pickle.dump(keys_list, file_handle)

## remove positions where too many sequences have gaps
pos_idx = []
for i in range(seq_msa.shape[1]):
    if np.sum(seq_msa[:,i] == 0) <= seq_msa.shape[0]*0.2:
        pos_idx.append(i)
with open("./output/" + "/seq_pos_idx.pkl", 'wb') as file_handle:
    pickle.dump(pos_idx, file_handle)
    
seq_msa = seq_msa[:, np.array(pos_idx)]
with open("./output/" + "/seq_msa.pkl", 'wb') as file_handle:
    pickle.dump(seq_msa, file_handle)

## reweighting sequences
seq_weight = np.zeros(seq_msa.shape)
for j in range(seq_msa.shape[1]):
    aa_type, aa_counts = np.unique(seq_msa[:,j], return_counts = True)
    num_type = len(aa_type)
    aa_dict = {}
    for a in aa_type:
        aa_dict[a] = aa_counts[list(aa_type).index(a)]
    for i in range(seq_msa.shape[0]):
        seq_weight[i,j] = (1.0/num_type) * (1.0/aa_dict[seq_msa[i,j]])
tot_weight = np.sum(seq_weight)
seq_weight = seq_weight.sum(1) / tot_weight 
with open("./output/" + "/seq_weight.pkl", 'wb') as file_handle:
    pickle.dump(seq_weight, file_handle)

## change aa numbering into binary
K = 21 ## num of classes of aa
D = np.identity(K)
num_seq = seq_msa.shape[0]
len_seq_msa = seq_msa.shape[1]
seq_msa_binary = np.zeros((num_seq, len_seq_msa, K))
for i in range(num_seq):
    seq_msa_binary[i,:,:] = D[seq_msa[i]]

with open("./output/" + "/seq_msa_binary.pkl", 'wb') as file_handle:
    pickle.dump(seq_msa_binary, file_handle)
