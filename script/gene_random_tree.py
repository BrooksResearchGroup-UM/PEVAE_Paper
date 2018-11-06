__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/02/09 14:11:14"

import random
import numpy as np
from ete3 import Tree

t = Tree()
random.seed(0)
num_leaf_nodes = 10000
t.populate(num_leaf_nodes,
           names_library = range(num_leaf_nodes),
           random_branches = True,
           branch_range = (0, 0.3))

idx_nodes = num_leaf_nodes
for node in t.traverse('preorder'):
    if node.name == "":
        node.name = str(idx_nodes)
        idx_nodes += 1

t.write(outfile = "./output/random_tree.newick", format = 1)
