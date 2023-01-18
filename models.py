import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy

class Text_validator(nn.Module):
    def __init__(self, embedding_dim, ngram_len, hidden_units, n_hidden_layers):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ngram_len = ngram_len

        self.input_to_hidden1 = nn.Linear(in_features=ngram_len*embedding_dim,
                                          out_features=hidden_units)
        self.hidden_layer_stack = [nn.Linear(in_features=hidden_units, out_features=hidden_units) for _ in range(n_hidden_layers)]

        self.hidden_to_output = nn.Linear(in_features=hidden_units, out_features=1)

    def forward(self, word_embeddings):
        X = self.input_to_hidden1(word_embeddings)
        '''
            Todo
        '''