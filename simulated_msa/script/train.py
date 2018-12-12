import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sys import exit
import sys
from VAE_model import *

## read data
with open("./output/msa_leaf_binary.pkl", 'rb') as file_handle:
    msa_binary = pickle.load(file_handle)    
num_seq = msa_binary.shape[0]
len_protein = msa_binary.shape[1]
num_res_type = msa_binary.shape[2]
msa_binary = msa_binary.reshape((num_seq, -1))
msa_binary = msa_binary.astype(np.float32)

with open("./output/msa_leaf_keys.pkl", 'rb') as file_handle:
    msa_keys = pickle.load(file_handle)    

msa_weight = np.ones(num_seq) / num_seq
msa_weight = msa_weight.astype(np.float32)

batch_size = num_seq
train_data = MSA_Dataset(msa_binary, msa_weight, msa_keys)
train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
vae = VAE(20, 2, len_protein * num_res_type, 100)
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
        msa = Variable(msa).cuda()
        weight = Variable(weight).cuda()
        
        optimizer.zero_grad()
        
        mu, sigma, p = vae.forward(msa)
        
        loss = loss_function(msa, weight, mu, sigma, p)
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
