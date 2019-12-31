__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/02/08 15:42:32"

'''
To simulate protein evolution, we need an amino-acid replacement matrix that specifies
the relative mutation rate between pairs of amino acids. Here we used the LG amino acid
replacement matrix. See (https://www.ncbi.nlm.nih.gov/pubmed/18367465) 
'''

import numpy as np
import pickle

## read the LG replacement matrix into a dictionary
## R is the transition rate matrix
## PI is a vector representing the equlibrium distributions
## of LG replacement matrix
amino_acids = ['A', 'R', 'N', 'D', 'C',
               'Q', 'E', 'G', 'H', 'I',
               'L', 'K', 'M', 'F', 'P',
               'S', 'T', 'W', 'Y', 'V']
R_dict = {}
PI_dict = {}
with open("./script/lg_LG.PAML.txt", 'r') as file_handle:
    line_num = 0
    for line in file_handle:
        line = line.strip()
        fields = line.split()
        if line_num < 20:
            assert(len(fields) == line_num)
            if len(fields) != 0:
                for i in range(line_num):
                    R_dict[(amino_acids[line_num], amino_acids[i])] = fields[i]
        else:
            if len(fields) != 0:
                for i in range(len(fields)):
                    PI_dict[amino_acids[i]] = fields[i]
        line_num += 1

## convert dictionaries into matrices
R = np.zeros((20,20))        
PI = np.zeros((20, 1))
for i in range(len(amino_acids)):
    for j in range(len(amino_acids)):        
        if i > j:
            R[i,j] = float(R_dict[(amino_acids[i], amino_acids[j])])
            R[j,i] = R[i,j]
    PI[i] = PI_dict[amino_acids[i]]


## calculate Q matrix     
Q = np.zeros((20,20))
for i in range(20):
    for j in range(20):
        if i != j:
            Q[i,j] = PI[j] * R[i,j]
    Q[i,i] = - np.sum(Q[i,:])

Q_dict = {}
for i in range(20):
    for j in range(20):
        Q_dict[(amino_acids[i], amino_acids[j])] = Q[i,j]

## save the LG replacement matrix        
with open("./output/LG_matrix.pkl", 'wb') as file_handle:
    pickle.dump({'R': R, 'Q': Q, 'PI': PI,
                 'R_dict': R_dict, 'Q_dict': Q_dict,
                 'PI_dict': PI_dict, 'amino_acids': amino_acids}, file = file_handle)
