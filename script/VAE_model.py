__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/10/16 02:50:08"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class MSA_Dataset(Dataset):
    def __init__(self, seq_msa_binary, seq_weight, seq_keys):
        super(MSA_Dataset).__init__()
        self.seq_msa_binary = seq_msa_binary
        self.seq_weight = seq_weight
        self.seq_keys = seq_keys
    def __len__(self):
        assert(self.seq_msa_binary.shape[0] == len(self.seq_weight))
        assert(self.seq_msa_binary.shape[0] == len(self.seq_keys))        
        return self.seq_msa_binary.shape[0]
    def __getitem__(self, idx):
        return self.seq_msa_binary[idx, :], self.seq_weight[idx], self.seq_keys[idx]

class VAE(nn.Module):
    def __init__(self, num_aa_type, dim_latent_vars, dim_msa_vars, num_hidden_units):
        super(VAE, self).__init__()

        self.num_aa_type = num_aa_type
        self.dim_latent_vars = dim_latent_vars
        self.dim_msa_vars = dim_msa_vars
        self.num_hidden_units = num_hidden_units
        
        self.encoder_fc1 = nn.Linear(dim_msa_vars, num_hidden_units)
        self.encoder_fc2_mu = nn.Linear(num_hidden_units, dim_latent_vars, bias = True)
        self.encoder_fc2_logsigma2 = nn.Linear(num_hidden_units, dim_latent_vars, bias = True)

        self.decoder_fc1 = nn.Linear(dim_latent_vars, num_hidden_units, bias = True)
        self.decoder_fc2 = nn.Linear(num_hidden_units, dim_msa_vars, bias = True)

    def encoder(self, x):
        h = self.encoder_fc1(x)
        h = torch.tanh(h)
        mu = self.encoder_fc2_mu(h)
        logsigma2 = self.encoder_fc2_logsigma2(h)
        sigma = torch.sqrt(torch.exp(logsigma2))
        return mu, sigma

    def sample_latent_var(self, mu, sigma):
        eps = Variable(sigma.data.new(sigma.size()).normal_())
        z = mu + sigma * eps
        return z

    def decoder(self, z):
        h = self.decoder_fc1(z)
        h = torch.tanh(h)
        h = self.decoder_fc2(h)
        h = h.view(h.size(0), -1, self.num_aa_type)
        p = F.log_softmax(h, dim = 2)
        p = p.view(p.size(0), -1)
        return p
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.sample_latent_var(mu, sigma)
        p = self.decoder(z)        
        return mu, sigma, p
    
def loss_function(msa, weight, mu, sigma, p):    
    cross_entropy = -torch.sum(torch.sum(msa*p, dim = 1) * weight)    
    KLD = - 0.5 * torch.sum(torch.sum((1.0 + torch.log(sigma**2) - mu**2 - sigma**2), dim = 1) * weight)
    return cross_entropy + KLD
