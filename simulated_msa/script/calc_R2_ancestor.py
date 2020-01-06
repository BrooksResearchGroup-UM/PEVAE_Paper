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

## read latent space representation
with open("./output/latent_space.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
key = data['key']
mu = data['mu']
sigma = data['sigma']

key2idx = {}
for i in range(len(key)):
    key2idx[key[i]] = i

## read tree
t = Tree("./output/random_tree.newick", format = 1)
num_leaf = len(t)
t.name = str(num_leaf)
leaf_idx = []
ancestral_idx = []
for i in range(len(key)):
    if int(key[i]) < num_leaf:
        leaf_idx.append(i)
    else:
        ancestral_idx.append(i)
        
for node in t.traverse('preorder'):
    if node.is_root():
        node.add_feature('anc', [])
        node.add_feature('sumdist', 0)
    else:
        node.add_feature('anc', node.up.anc + [node.up.name])
        node.add_feature('sumdist', node.up.sumdist + node.dist)
        
reg = linear_model.LinearRegression()
pca = PCA(n_components = 2)
leaf = [str(i) for i in range(num_leaf)]
R2 = []
PCC = []
PCC_naive = []
num_anc_list = []
for k in range(len(leaf)):
    print(k, flush = True)

    leaf_name = leaf[k]
    idx = key.index(leaf_name)
    data = pd.DataFrame(index = (t&leaf_name).anc + [leaf_name], columns = ("mu1", 'mu2', 'depth'))
    data.loc[leaf_name, :] = (mu[idx, 0], mu[idx, 1], (t&leaf_name).sumdist)
    num_anc = len((t&leaf_name).anc)    
    num_anc_list.append(num_anc)
    
    for i in range(num_anc):
        n = (t&leaf_name).anc[i]
        idx = key2idx[n]
        data.loc[n, :] = (mu[idx, 0], mu[idx, 1], (t&n).sumdist)

    data = np.array(data).astype(np.float64)
    res = reg.fit(data[:,0:2], data[:,-1])
    yhat = res.predict(data[:,0:2])    
    
    SS_res = np.sum((data[:,-1] - yhat)**2)
    SS_tot = np.sum((data[:,-1] - np.mean(data[:,-1]))**2)
    r2 = 1 - SS_res/SS_tot
    R2.append(r2)

    pca.fit(data[:,0:2])
    pca_coor = pca.transform(data[:,0:2])

    if np.sum(pca.components_[0,:] * data[-1,0:2]) < 0:
        main_coor = -pca_coor[:,0]
    else:
        main_coor = pca_coor[:,0]
    PCC.append(stats.pearsonr(main_coor, data[:,2])[0])

    direction = data[-1,0:2]/np.sqrt(data[-1,0:2]**2)
    naive_coor = np.sum(data[:,0:2] * direction, 1)
    PCC_naive.append(stats.pearsonr(naive_coor, data[:,2])[0])
            
with open("./output/R2.pkl", 'wb') as file_handle:
    pickle.dump({'leaf_name': leaf, 'R2': R2,
                 'num_anc': num_anc_list, 'PCC': PCC,
                 'PCC_naive': PCC_naive}, file = file_handle)
