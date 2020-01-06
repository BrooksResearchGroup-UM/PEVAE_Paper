import matplotlib as mpl
mpl.use("Agg")
mpl.rc('font', size = 14)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import pickle
import torch
import pandas
import sys
from VAE_model import *
from torch.utils.data import Dataset, DataLoader
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sys import exit
from ete3 import Tree

## read data
with open("./output/msa_binary.pkl", 'rb') as file_handle:
    msa_binary = pickle.load(file_handle)    
num_seq = msa_binary.shape[0]
len_protein = msa_binary.shape[1]
num_res_type = msa_binary.shape[2]
msa_binary = msa_binary.reshape((num_seq, -1))
msa_binary = msa_binary.astype(np.float32)

with open("./output/msa_keys.pkl", 'rb') as file_handle:
    msa_keys = pickle.load(file_handle)    

msa_weight = np.ones(num_seq) / num_seq
msa_weight = msa_weight.astype(np.float32)

batch_size = 128
train_data = MSA_Dataset(msa_binary, msa_weight, msa_keys)
train_data_loader = DataLoader(train_data, batch_size = batch_size)
vae = VAE(20, 2, len_protein * num_res_type, [100])
vae.cuda()
vae.load_state_dict(torch.load("./output/vae_0.01.model"))

elbo_list = []
for idx, data in enumerate(train_data_loader):
    print(idx)
    msa, weight, key = data
    with torch.no_grad():
        msa = msa.cuda()
        elbo = vae.compute_elbo_with_multiple_samples(msa, 5000)
        elbo_list.append(elbo.cpu().data.numpy())

elbo = np.concatenate(elbo_list)
with open("./output/elbo.pkl", 'wb') as file_handle:
    pickle.dump(elbo, file_handle)    
