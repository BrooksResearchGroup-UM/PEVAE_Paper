# Protein Evolution with Variational Auto-Encoder (PEVAE)

## Required Python packages
pytorch, numpy, ete3

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


