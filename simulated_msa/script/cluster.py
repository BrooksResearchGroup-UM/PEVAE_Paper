__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/02/17 21:29:29"

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
import time

## read latent space representation
with open("./output/latent_space.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
key = data['key']
mu = data['mu']
sigma = data['sigma']
p = data['p']

key2idx = {}
for i in range(len(key)):
    key2idx[key[i]] = i

## read tree
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
        
for node in t.traverse('preorder'):
    if node.is_root():
        node.add_feature('anc', [])
        node.add_feature('sumdist', 0)
    else:
        node.add_feature('anc', node.up.anc + [node.up.name])
        node.add_feature('sumdist', node.up.sumdist + node.dist)

color_names = list(mpl.colors.cnames.keys())
dist_cutoff = 0.5
head_node_names = []
for node in t.traverse('preorder'):
    if node.is_leaf() and node.sumdist < dist_cutoff:
        head_node_names.append(node.name)
    if (not node.is_leaf()) and node.sumdist > dist_cutoff and node.up.sumdist < dist_cutoff:
        head_node_names.append(node.name)

cluster_node_names = {}
for name in head_node_names:
    cluster_node_names[name] = []
    for node in (t&name).traverse('preorder'):
        cluster_node_names[name].append(node.name)

fig = plt.figure(0)
fig.clf()
for i in range(len(head_node_names)):
    names = cluster_node_names[head_node_names[i]]
    idx = [ key2idx[n] for n in names]
    plt.plot(mu[idx,0], mu[idx,1], '.', markersize = 2, label = head_node_names[i])
# plt.xlim((-6.5,10))
# plt.ylim((-8,8))    
# plt.title("dist_cutoff: {:.2f}, num of cluster: {:}".format(dist_cutoff, len(head_node_names)))
# plt.legend(markerscale= 4, loc = 'upper left')
plt.xlabel(r'$Z_1$')
plt.ylabel(r'$Z_2$')
plt.tight_layout()
fig.savefig("./output/cluster.eps")

## zoom in branches
for branch in ['10854', '16528']:
    sub_t = t&branch
    dist_cutoff = sub_t.sumdist + 0.3
    head_node_names = []
    for node in sub_t.traverse('preorder'):
        if node.is_leaf() and node.sumdist < dist_cutoff:
            head_node_names.append(node.name)
        if (not node.is_leaf()) and node.sumdist > dist_cutoff and node.up.sumdist < dist_cutoff:
            head_node_names.append(node.name)

    cluster_node_names = {}
    for name in head_node_names:
        cluster_node_names[name] = []
        for node in (t&name).traverse('preorder'):
            cluster_node_names[name].append(node.name)

    fig = plt.figure(1)
    fig.clf()
    for i in range(len(head_node_names)):
        names = cluster_node_names[head_node_names[i]]
        idx = [ key2idx[n] for n in names]
        plt.plot(mu[idx,0], mu[idx,1], '.', markersize = 2, label = head_node_names[i])
    # plt.xlim((-6.5,10))
    # plt.ylim((-8,8))    
    # plt.title("dist_cutoff: {:.2f}, num of cluster: {:}".format(dist_cutoff, len(head_node_names)))
    # plt.legend(markerscale= 4, loc = 'upper right')
    plt.xlabel(r'$Z_1$')
    plt.ylabel(r'$Z_2$')
    plt.tight_layout()
    fig.savefig("./output/branch_{}_cluster.eps".format(branch))


#plt.show()


