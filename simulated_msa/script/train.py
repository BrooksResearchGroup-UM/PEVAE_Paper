"""
Train a latent space model using leaf node sequences using VAEs
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sys import exit
import sys
sys.path.append("./script/")
from VAE_model import *

## read multiple sequence alignment in binary representation
with open("./output/msa_leaf_binary.pkl", 'rb') as file_handle:
    msa_binary = pickle.load(file_handle)    
num_seq = msa_binary.shape[0]
len_protein = msa_binary.shape[1]
num_res_type = msa_binary.shape[2]
msa_binary = msa_binary.reshape((num_seq, -1))
msa_binary = msa_binary.astype(np.float32)

## each sequence has a label
with open("./output/msa_leaf_keys.pkl", 'rb') as file_handle:
    msa_keys = pickle.load(file_handle)    

## sequences in msa are weighted. Here sequences are assigned
## the same weights
msa_weight = np.ones(num_seq) / num_seq
msa_weight = msa_weight.astype(np.float32)

#### construct the MSA dataseta and DataLoader
## the model is trained with batches of data.
## here we choose the batch size to be equal to the total number
## of sequences because the GPU usually has large enough memory
## for it. If you are running the code on a GPU with small memory and
## have GPU memory error, try to decrease the batch_size
batch_size = num_seq
train_data = MSA_Dataset(msa_binary, msa_weight, msa_keys)
train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

## here we use only one hidden layer with 100 neurons. If you want to use more
## hidden layers, change the parameter num_hidden_units. For instance, changing
## it to [100, 150] will use two hidden layers with 100 and 150 neurons.
vae = VAE(num_aa_type = 20,
          dim_latent_vars = 2,
          dim_msa_vars = len_protein*num_res_type,
          num_hidden_units = [100])
vae.cuda()

weight_decay = 0.01
optimizer = optim.Adam(vae.parameters(), weight_decay = 0.01)
num_epoches = 10000
train_loss_epoch = []
test_loss_epoch = []

for epoch in range(num_epoches):
    running_loss = []    
    for idx, data in enumerate(train_data_loader):
        msa, weight, _ = data
        msa = msa.cuda()
        weight = weight.cuda()

        loss = (-1)*vae.compute_weighted_elbo(msa, weight)
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()
        
        print("Epoch: {:>4}, Step: {:>4}, loss: {:>4.2f}".format(epoch, idx, loss.data.item()), flush = True)
        running_loss.append(loss.data.item())        
    train_loss_epoch.append(np.mean(running_loss))

torch.save(vae.state_dict(), "./output/vae_{:.2f}.model".format(weight_decay))

with open('./output/loss.pkl', 'wb') as file_handle:
    pickle.dump({'train_loss_epoch': train_loss_epoch}, file_handle)

fig = plt.figure(0)
fig.clf()
plt.plot(train_loss_epoch, label = "train", color = 'r')
#plt.plot(test_loss_epoch, label = "test", color = 'b')
#plt.ylim((140, 180))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title("Loss")
fig.savefig("./output/loss.png")
#plt.show()
