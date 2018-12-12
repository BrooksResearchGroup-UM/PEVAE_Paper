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

batch_size = num_seq
train_data = MSA_Dataset(msa_binary, msa_weight, msa_keys)
train_data_loader = DataLoader(train_data, batch_size = batch_size)
vae = VAE(20, 2, len_protein * num_res_type, 100)
vae.cuda()
vae.load_state_dict(torch.load("./output/vae_0.01.model"))

for idx, data in enumerate(train_data_loader):
    msa, weight, key = data
    with torch.no_grad():
        msa = Variable(msa).cuda()
        mu, sigma, p = vae.forward(msa)

mu = mu.cpu().data.numpy()
sigma = sigma.cpu().data.numpy()
p = p.cpu().data.numpy()

with open("./output/latent_space.pkl", 'wb') as file_handle:
    pickle.dump({'key': key, 'mu': mu, 'sigma': sigma, 'p': p}, file_handle)    

## plot latent space    
t = Tree("./output/named_tree.newick", format = 1)
num_leaf = len(t)
t.name = str(num_leaf)

leaf_idx = []
ancestral_idx = []
for i in range(len(key)):
    if int(key[i]) < num_leaf:
        leaf_idx.append(i)
    else:
        ancestral_idx.append(i)

plt.figure(0)
plt.clf()
plt.plot(mu[leaf_idx,0], mu[leaf_idx,1], 'b.', alpha = 0.5, markersize = 2)
plt.xlim((-6.5,6.5))
plt.ylim((-6.5,6.5))
plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.tight_layout()
plt.savefig("./output/latent_mu_leaf.pdf")

plt.figure(1)
plt.clf()
plt.plot(mu[ancestral_idx,0], mu[ancestral_idx,1], 'r.', alpha = 0.5, markersize = 2)
plt.xlim((-6.5,6.5))
plt.ylim((-6.5,6.5))
plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.tight_layout()
plt.savefig("./output/latent_mu_ancestral.pdf")

plt.figure(2)
plt.clf()
plt.plot(mu[ancestral_idx,0], mu[ancestral_idx,1], 'r.', alpha = 0.5, markersize = 1, label = 'ancestral')
plt.plot(mu[leaf_idx,0], mu[leaf_idx,1], 'b.', alpha = 0.5, markersize = 1, label = 'leaf')
#plt.plot(mu[:,0], mu[:,1], 'c.', alpha = 0.5, markersize = 1)
plt.xlim((-6.5,6.5))
plt.ylim((-6.5,6.5))
plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.legend(markerscale = 3)
plt.tight_layout()
plt.savefig("./output/latent_mu_all.pdf")

    
