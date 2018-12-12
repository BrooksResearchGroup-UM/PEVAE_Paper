__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/02/10 19:26:02"

import pickle
from ete3 import Tree
from sys import exit
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rc('font', size = 14)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.stats as stats
from sklearn.decomposition import PCA
import time
        
with open("./output/R2.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)

leaf_name = data['leaf_name']
R2 = data['R2']
num_anc = data['num_anc']
PCC = data['PCC']
PCC_naive = data['PCC_naive']

## read latent space representation
with open("./output/latent_space.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
key = data['key']
mu = data['mu']
sigma = data['sigma']
p = data['p']

key2idx = {}
for i in range(len(key)):
    key2idx[key[i]] = i

## plot
# ## coefficient of determination
# fig = plt.figure(0)
# fig.clf()
# plt.hist(R2, 25, normed = True)
# plt.title("coefficient of determination R^2")
# fig.savefig("./output/R2.pdf")

# fig = plt.figure(1)
# fig.clf()
# plt.hist(np.sqrt(R2), 25, normed = True)
# plt.title("coefficient of determination R")
# fig.savefig("./output/R.pdf")

fig = plt.figure(1)
fig.clf()
plt.hist(PCC, 25, normed = True)
#plt.title("pearson correlation coefficient r")
#plt.xlim((0,1))
plt.xlabel('Pearson correlation coefficient')
plt.tight_layout()
fig.savefig("./output/PCC.eps")

# fig = plt.figure(3)
# fig.clf()
# tmp = [PCC[i] for i in range(len(PCC)) if np.sqrt(np.sum(mu[key.index(leaf_name[i]), :]**2)) >= 0.5]
# plt.hist(tmp, 25, normed = True)
# plt.title("pearson correlation coefficient r")
# fig.savefig("./output/PCC_dist_cutoff_0.5.pdf")

# fig = plt.figure(4)
# fig.clf()
# plt.hist(PCC_naive, 25, normed = True)
# plt.title("pearson correlation coefficient r")
# fig.savefig("./output/PCC_naive.pdf")

# fig = plt.figure(5)
# fig.clf()
# tmp = [PCC_naive[i] for i in range(len(PCC_naive)) if np.sqrt(np.sum(mu[key.index(leaf_name[i]), :]**2)) >= 0.5]
# plt.hist(tmp, 25, normed = True)
# plt.title("pearson correlation coefficient r")
# fig.savefig("./output/PCC_naive_dist_cutoff_0.5.pdf")

exit()
