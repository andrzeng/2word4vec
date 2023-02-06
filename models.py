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

class missingWordPredictor(nn.Module):
    def __init__(
        self,
        num_embeddings : int,
        embedding_dim : int,
        
        phrase_len : int = 7
    ):
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.phrase_len = phrase_len
        super(missingWordPredictor, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.linear1 = nn.Linear(in_features=(phrase_len-1)*embedding_dim, out_features=100)
        self.linear2 = nn.Linear(in_features=100, out_features=100)
        self.linear3 = nn.Linear(in_features=100, out_features=embedding_dim)

    def forward(self, 
    X : torch.LongTensor #LongTensor of lookup table indices corresponding to the 7 words in the phrase.
    ):
        #X = torch.cat([X[:,:int((self.phrase_len-1)/2)], X[:,int((self.phrase_len+1)/2):]], dim=1) #Remove the middle item

        X = self.embedding(X) #Convert indices into embedding tensors
        X = torch.flatten(X, start_dim=1)
        
        X = self.linear1(X)
        X = torch.relu(X)
        X = self.linear2(X)
        X = torch.relu(X)
        X = self.linear3(X)

        return X
    
    def get_embedding(self, X: torch.LongTensor):
        return self.embedding(X)

