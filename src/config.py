from imports import *

BATCH_SIZE = 20
n = 5 # number of negative samples to sample for each positive sample
n_epochs = 50
cons_dims = 0
embed_dim = 32
depth = 2
n_heads = 4
att_dropout = 0.5
an_dropout = 0.5
ffn_dropout = 0.5
mlp_dims = [16, 16]
learning_rate = 1e-4
weight_decay = 0.004
pos_weight_power = 6