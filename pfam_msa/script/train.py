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
sys.path.append("/home/xqding/course/projectsOnGitHub/PEVAE_Paper/pfam_msa/script")
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

#### training model with K-fold cross validation
## split the data index 0:num_seq-1 into K sets
## each set is just a set of indices of sequences.
## in the kth traing, the kth subsets of sequences are used
## as validation data and the remaining K-1 sets are used
## as training data
K = 5
num_seq_subset = num_seq // K + 1
idx_subset = []
random_idx = np.random.permutation(range(num_seq))
for i in range(K):
    idx_subset.append(random_idx[i*num_seq_subset:(i+1)*num_seq_subset])

## the following list holds the elbo values on validation data    
elbo_all_list = []

for k in range(K):
    print("Start the {}th fold training".format(k))
    print("-"*60)
    
    ## build a VAE model with random parameters
    vae = VAE(21, 2, len_protein * num_res_type, [100])

    ## move the VAE onto a GPU
    vae.cuda()

    ## build the Adam optimizer
    optimizer = optim.Adam(vae.parameters(),
                           weight_decay = weight_decay)

    ## collect training and valiation data indices
    validation_idx = idx_subset[k]
    validation_idx.sort()
    
    train_idx = np.array(list(set(range(num_seq)) - set(validation_idx)))
    train_idx.sort()

    train_msa = torch.from_numpy(seq_msa_binary[train_idx, ])
    train_msa = train_msa.cuda()

    train_weight = torch.from_numpy(seq_weight[train_idx])
    train_weight = train_weight/torch.sum(train_weight)
    train_weight = train_weight.cuda()
    
    train_loss_list = []    
    for epoch in range(num_epoches):    
        loss = (-1)*vae.compute_weighted_elbo(train_msa, train_weight)
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()

        train_loss_list.append(loss.item())
        if (epoch + 1) % 50 ==0:
            print("Fold: {}, Epoch: {:>4}, loss: {:>4.2f}".format(k, epoch, loss.item()), flush = True)

    ## cope trained model to cpu and save it
    vae.cpu()
    torch.save(vae.state_dict(), "./output/model/vae_{}_fold_{}.model".format(str(weight_decay), k))

    print("Finish the {}th fold training".format(k))
    print("="*60)
    print('')
    
    print("Start the {}th fold validation".format(k))
    print("-"*60)
    ## evaluate the trained model 
    vae.cuda()
    
    elbo_on_validation_data_list = []
    ## because the function vae.compute_elbo_with_multiple samples uses
    ## a large amount of memory on GPUs. we have to split validation data
    ## into batches.
    batch_size = 128
    num_batches = len(validation_idx)//batch_size + 1
    for idx_batch in range(num_batches):
        if (idx_batch + 1) % 50 == 0:
            print("idx_batch: {} out of {}".format(idx_batch, num_batches))        
        validation_msa = seq_msa_binary[validation_idx[idx_batch*batch_size:(idx_batch+1)*batch_size]]
        validation_msa = torch.from_numpy(validation_msa)
        with torch.no_grad():
            validation_msa = validation_msa.cuda()
            elbo = vae.compute_elbo_with_multiple_samples(validation_msa, 5000)            
            elbo_on_validation_data_list.append(elbo.cpu().data.numpy())

    elbo_on_validation_data = np.concatenate(elbo_on_validation_data_list)    
    elbo_all_list.append(elbo_on_validation_data)
    
    print("Finish the {}th fold validation".format(k))
    print("="*60)
    
elbo_all = np.concatenate(elbo_all_list)
elbo_mean = np.mean(elbo_all)
## the mean_elbo can approximate the quanlity of the learned model
## we want a model that has high mean_elbo
print("mean_elbo: {:.3f}".format(elbo_mean))

with open("./output/elbo_all.pkl", 'wb') as file_handle:
    pickle.dump(elbo_all, file_handle)

    
exit()
