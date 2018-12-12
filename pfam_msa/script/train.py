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
from model import *
from sys import exit
import argparse

parser = argparse.ArgumentParser(description='Parameters for training the model')
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--weight_decay', type = float)
args = parser.parse_args()

num_epoches = args.num_epoch
weight_decay = args.weight_decay

print("num_epoches: ", num_epoches)
print("weight_decay: ", str(weight_decay))

## read data
with open("./output/seq_msa_binary.pkl", 'rb') as file_handle:
    seq_msa_binary = pickle.load(file_handle)    
num_seq = seq_msa_binary.shape[0]
len_protein = seq_msa_binary.shape[1]
num_res_type = seq_msa_binary.shape[2]
seq_msa_binary = seq_msa_binary.reshape((num_seq, -1))
seq_msa_binary = seq_msa_binary.astype(np.float32)

with open("./output/seq_weight.pkl", 'rb') as file_handle:
    seq_weight = pickle.load(file_handle)
seq_weight = seq_weight.astype(np.float32)

batch_size = num_seq
train_data = MSA_Dataset(seq_msa_binary, seq_weight)
train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
vae = VAE(2, len_protein * num_res_type, 100)
vae.cuda()

optimizer = optim.Adam(vae.parameters(), weight_decay = weight_decay)
train_loss_epoch = []
test_loss_epoch = []

for epoch in range(num_epoches):
    running_loss = []    
    for idx, data in enumerate(train_data_loader):
        msa, weight = data
        msa = Variable(msa).cuda()
        weight = Variable(weight).cuda()
        
        optimizer.zero_grad()
        
        mu, sigma, p = vae.forward(msa)
        
        loss = loss_function(msa, weight, mu, sigma, p)
        loss.backward()
        optimizer.step()    
        print("Epoch: {:>4}, Step: {:>4}, loss: {:>4.2f}".format(epoch, idx, loss.item()))
        running_loss.append(loss.item())        
    train_loss_epoch.append(np.mean(running_loss))
        
torch.save(vae.state_dict(), "./output/model/vae_{}.model".format(str(weight_decay)))

with open('./output/loss/loss_{}.pkl'.format(str(weight_decay)), 'wb') as file_handle:
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
fig.savefig("./output/loss/loss_{}.png".format(str(weight_decay)))
#plt.show()
