import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import pickle
import torch
import pandas
from VAE_model import *
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sys import exit

mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')

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
train_data_loader = DataLoader(train_data, batch_size = batch_size)
vae = VAE(2, len_protein * num_res_type, 100)
vae.cuda()
vae.load_state_dict(torch.load("./output/model/vae_0.01.model"))

for idx, data in enumerate(train_data_loader):
    msa, weight = data
    with torch.no_grad():
        msa = Variable(msa).cuda()    
        mu, sigma, p = vae.forward(msa)

mu = mu.cpu().data.numpy()
sigma = sigma.cpu().data.numpy()
p = p.cpu().data.numpy()

with open("./output/latent_space.pkl", 'wb') as file_handle:
    pickle.dump({'mu':mu, 'sigma': sigma, 'p': p}, file_handle)

with open("./output/latent_space.pkl", 'rb') as file_handle:
    latent_space = pickle.load(file_handle)
mu = latent_space['mu']
sigma = latent_space['sigma']
p = latent_space['p']
    
plt.figure(0)
plt.clf()
plt.plot(mu[:,0], mu[:,1], '.', alpha = 0.1, markersize = 3)
plt.xlim((-6,6))
plt.ylim((-6,6))
plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.tight_layout()
plt.savefig("./output/Fibronectin_III_latent_mu_scatter.png")
