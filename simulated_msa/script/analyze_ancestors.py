__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/02/10 19:26:02"

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
import scipy.stats as stats
from sklearn.decomposition import PCA

## read latent space representation
with open("./output/latent_space.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
key = data['key']
mu = data['mu']
sigma = data['sigma']

## read tree
t = Tree("./output/random_tree.newick", format = 1)
num_leaf = len(t)
t.name = str(num_leaf)
leaf_idx = []
ancestral_idx = []
for i in range(len(key)):
    if int(key[i]) < num_leaf:
        leaf_idx.append(i)
    else:
        ancestral_idx.append(i)

# leaf_mu = mu[leaf_idx, :]
# leaf_key = np.array(key)[leaf_idx]
# flag = (leaf_mu[:, 0] < -4) * (leaf_mu[:,1] > 2) * (leaf_mu[:,1] < 4)
# leaf_mu = leaf_mu[flag]
# leaf_key = leaf_key[flag]
# idx = leaf_key[np.argmin(leaf_mu[:,0])]

leaf = ['9747', '1848', '3790', '7238']

colormap = ['Greys', 'Blues', 'Reds', 'Purples']
anc_names = {}

for n in leaf:
    anc_names[n] = []
    ancestors = (t&n).get_ancestors()
    for anc in ancestors:
        anc_names[n].append(anc.name)

fig = plt.figure(1)
fig.clf()
for k in range(len(leaf)):
    leaf_name = leaf[k]
    idx = key.index(leaf_name)
    data = pd.DataFrame(index = [leaf_name] + anc_names[leaf_name], columns = ("mu1", 'mu2', 'depth'))
    data.loc[leaf_name, :] = (mu[idx, 0], mu[idx, 1], t.get_distance(t&leaf_name))
    num_anc = len(anc_names[leaf_name])
    for i in range(num_anc):
        n = anc_names[leaf_name][i]
        idx = key.index(n)
        data.loc[n, :] = (mu[idx, 0], mu[idx, 1], t.get_distance(t&n))

    plt.scatter(data.loc[:,'mu1'], data.loc[:,'mu2'], c = data.loc[:, 'depth'], cmap = plt.get_cmap('viridis'))
    plt.plot(data.loc[leaf_name,'mu1'], data.loc[leaf_name,'mu2'], '+r', markersize = 16)

# plt.xlim((-6.5,6.5))
# plt.ylim((-6.5,6.5))
plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.colorbar()
plt.tight_layout()
plt.savefig("./output/leaf_ancestry_latent_space.eps")

PCC = []
pca = PCA(n_components = 2)
reg = linear_model.LinearRegression()
fig = plt.figure(2)
fig.clf()
#for k in range(len(leaf)):
for k in [1]:
    leaf_name = leaf[k]
    idx = key.index(leaf_name)
    data = pd.DataFrame(index = [leaf_name] + anc_names[leaf_name], columns = ("mu1", 'mu2', 'depth'))
    data.loc[leaf_name, :] = (mu[idx, 0], mu[idx, 1], t.get_distance(t&leaf_name))
    num_anc = len(anc_names[leaf_name])
    for i in range(num_anc):
        n = anc_names[leaf_name][i]
        idx = key.index(n)
        data.loc[n, :] = (mu[idx, 0], mu[idx, 1], t.get_distance(t&n))

    data = np.array(data).astype(np.float64)
    pca.fit(data[:,0:2])
    pca_coor = pca.transform(data[:,0:2])

    if np.sum(pca.components_[0,:] * data[0,0:2]) < 0:
        main_coor = -pca_coor[:,0]
    else:
        main_coor = pca_coor[:,0]

    plt.plot(main_coor, data[:,-1], 'k.', markersize = 10)
    PCC.append(stats.pearsonr(main_coor, data[:,2])[0])

    res = reg.fit(main_coor.reshape((-1,1)), data[:,-1])
    yhat = res.predict(main_coor.reshape((-1,1)))
    
    SS_res = np.sum((data[:,-1] - yhat)**2)
    SS_tot = np.sum((data[:,-1] - np.mean(data[:,-1]))**2)
    r2 = 1 - SS_res/SS_tot

    reg_x = np.linspace(-4,4,30)
    reg_y = reg.intercept_ + reg.coef_[0] * reg_x
    plt.plot(reg_x, reg_y, 'b')
    plt.text(-2,3, r"$R^2$ = {:.2f}".format(r2), size = 16)
    plt.xlabel("coordinate along first component")
    plt.ylabel("evoluationary distance")
    plt.tight_layout()
    plt.savefig("./output/scatter_main_coor_dist.eps")
    #plt.show()

for k in range(len(leaf)):
    leaf_name = leaf[k]
    idx = key.index(leaf_name)
    data = pd.DataFrame(index = [leaf_name] + anc_names[leaf_name], columns = ("mu1", 'mu2', 'depth'))
    data.loc[leaf_name, :] = (mu[idx, 0], mu[idx, 1], t.get_distance(t&leaf_name))
    num_anc = len(anc_names[leaf_name])
    for i in range(num_anc):
        n = anc_names[leaf_name][i]
        idx = key.index(n)
        data.loc[n, :] = (mu[idx, 0], mu[idx, 1], t.get_distance(t&n))

    fig = plt.figure(0, figsize = (8,6))
    fig.clf()        
    plt.scatter(data.loc[:,'mu1'], data.loc[:,'mu2'], c = data.loc[:, 'depth'], cmap = plt.get_cmap(colormap[k]))
    plt.plot(data.loc[leaf_name,'mu1'], data.loc[leaf_name,'mu2'], '+r', markersize = 16)    
    # plt.xlim((-6.5,6.5))
    # plt.ylim((-6.5,6.5))
    plt.xlabel("$Z_1$")
    plt.ylabel("$Z_2$")    
    plt.colorbar()
    plt.title(leaf_name)    
    plt.savefig("./output/leaf_{}.pdf".format(leaf_name))
