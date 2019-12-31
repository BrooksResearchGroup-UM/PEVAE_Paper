__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/02/09 14:11:14"

'''
Generate a random phylogenetic tree using the populate function from the ete3 package.
'''

import random
import numpy as np
from ete3 import Tree

## generate a random phylogenetic tree with 10000 leaf nodes
t = Tree()
random.seed(0)
num_leaf_nodes = 10000
t.populate(num_leaf_nodes,
           names_library = range(num_leaf_nodes),
           random_branches = True,
           branch_range = (0, 0.3))

## leaf nodes are named as indices from 0 to 9999.
## non-leaf nodes are named incrementally from 10000
idx_nodes = num_leaf_nodes
for node in t.traverse('preorder'):
    if node.name == "":
        node.name = str(idx_nodes)
        idx_nodes += 1

## the random phylogenetic tree is saved in newick format
## see https://en.wikipedia.org/wiki/Newick_format
t.write(outfile = "./output/random_tree.newick", format = 1)
