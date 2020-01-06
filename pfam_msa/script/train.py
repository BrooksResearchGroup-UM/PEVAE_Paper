#!/home/xqding/apps/miniconda3/bin/python

# Created by Xinqiang Ding (xqding@umich.edu)
# at 2020/01/05 20:51:53

#SBATCH --job-name=VAE
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --exclude=gollum[003-045]
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1


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
import sys
sys.path.append("/home/xqding/course/projectsOnGitHub/PEVAE_Paper")
from VAE_model import *
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

with open("./output/keys_list.pkl", 'rb') as file_handle:
    seq_keys = pickle.load(file_handle)

vae = VAE(21, 2, len_protein * num_res_type, [100])
vae.cuda()

optimizer = optim.Adam(vae.parameters(),
                       weight_decay = weight_decay)
train_loss_epoch = []
test_loss_epoch = []

msa = torch.from_numpy(seq_msa_binary)
msa = msa.cuda()
weight =  torch.from_numpy(seq_weight)
weight = weight.cuda()

loss_list = []
for epoch in range(num_epoches):    
    loss = (-1)*vae.compute_weighted_elbo(msa, weight)
    optimizer.zero_grad()
    loss.backward()        
    optimizer.step()

    loss_list.append(loss.item())
    print("Epoch: {:>4}, loss: {:>4.2f}".format(epoch, loss.item()), flush = True)
    
torch.save(vae.state_dict(), "./output/model/vae_{}.model".format(str(weight_decay)))

with open('./output/loss/loss_{}.pkl'.format(str(weight_decay)), 'wb') as file_handle:
    pickle.dump({'loss_list': loss_list}, file_handle)

fig = plt.figure(0)
fig.clf()
plt.plot(loss_list, label = "train", color = 'r')
#plt.plot(test_loss_epoch, label = "test", color = 'b')
#plt.ylim((140, 180))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title("Loss")
fig.savefig("./output/loss/loss_{}.png".format(str(weight_decay)))
#plt.show()
