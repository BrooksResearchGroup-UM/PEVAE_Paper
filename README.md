# Protein Evolution with Variational Auto-Encoder (PEVAE)

## Requirements
Operation System: Linux (CentOS release 6.9)  
Programming language: Python 3.5  
Python Package: PyTorch 1.0.0, Numpy 1.15, ete2 3.1.1

**Note**: Although the souce code in this repository has only been tested on a Linux system,
it is expected to be able to run on other operation systems (Windows or Mac)
as long as both Python and required packages are installed. To install both Python and required
Python packages, we highly recommend readers to first install Anaconda and then install required
packages via `conda install`:  
```
conda install pytorch torchvision -c pytorch
conda install numpy
conda install ete3
```

## Installation
In order to run the following demo, the source code in this repository needs to be downloaded
onto your local computer. You can download all the source code via
`git clone https://github.com/xqding/PEVAE_Paper.git`.
If the command runs successfully, you should be able to find a new dictory called `PEVAE_Paper`
under the directory where you run the `git clone` command.
You can find all the source code in the directory `PEVAE_Paper`.


## Learn a VAE model on a simulated multiple sequence alignment
1. Simulate a multiple sequence alignment.

   Run `python ./script/gene_random_tree.py` to generate a random phylogenetic tree `./output/random_tree.newick`.

   Run `python ./script/read_LG_matrix.py` to generate the LG amino acid substitute matrix `./output/LG_matrix.pkl`

   Run `python ./script/simulate.py` to simulate a multiple sequence alignment `./output/simulated_msa.pkl` based on the phlogenetic tree and LG_matrix generated above.

2. Train a VAE model using the simulated multiple sequence alignment.

   Run `python ./script/proc_msa.py` to process the multiple sequence alignment into a binary matrix `./output/msa_binary.pkl`.

   Run `python ./script/train.py` to train a VAE model using the binary representation of the multiple sequence alignment.

3. Project sequences into the VAE latent space.

   Run `python ./script/analyze_model.py` to calculate the latent space coordinates of all the sequences in the multiple sequence alignment.

   Run `python ./script/cluster.py` to color sequences in the latent space based on their phlogenetic relationships

   Run `python ./script/analyze_ancestors.py` to ancestral relationship between sequences in latent space.

   Run `python ./script/calc_R2_ancestor.py` to calculate Pearson correlation coefficient between evolutionary time and positions of sequences in latent space.

   Run `python ./script/plot_R2.py` to plot the Pearson correlation coefficient calculated above.

## Learn a VAE model on a multiple sequence alignment of a protein family from Pfam
1. Download the multiple sequence alignment given a Pfam id

   You can either run `python ./script/download_MSA.py --Pfam_id PF00041` or go to Pfam website to download the sequence alignment.
   
2. Train a VAE model using the simulated multiple sequence alignment.
   Run `python ./script/proc_msa.py` to process the alignment
   
   Run `python ./script/train.py --num_epoch 10000 --weight_decay 0.01` to train a VAE model of the multiple sequence alignment.
   
3. Project sequences into the VAE latent space.
   
   Run `python ./script/analyze_model.py` to project sequences into latent space.
