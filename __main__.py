import vocabulary
import string
import models
import sys
import torch
import numpy as np
import numpy as np
import random
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


def train_word2vec_model(model : models.missingWordPredictor, 
                        loader: DataLoader,
                        loss_fn,
                        optimizer,
                        n_epochs: int=10,
                        ngram_length: int=7,
                        ):

    model.train()

    for epoch in range(n_epochs):
        total_loss = 0
        index = 0
        for X, y in loader:
            index += 1
            if(index % 1000 == 0):
                print(f"at checkpoint {index}/{len(loader)}")
            optimizer.zero_grad()

            loss = loss_fn(model(X), model.get_embedding(y))
            loss.backward()
            optimizer.step()
            #print(loss)
            total_loss += loss

        print(f"Average loss for epoch {epoch+1} is {total_loss/len(loader)}")


if __name__ == '__main__':
    '''Initialize array of the words in the Bible, in order'''
    text_bible = []
    with open('data/bible.txt','r') as f:
        for line in f:
            for word in line.split():
                text_bible.append(word.translate(str.maketrans('','', string.punctuation)).lower())           
    bible_lookup_table = vocabulary.create_lookup_table(text_bible)


    dataset = vocabulary.CorpusDataset("data/bible.txt", 7, bible_lookup_table)
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    
    
    model = models.missingWordPredictor(3000, 10)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_word2vec_model(model, loader, loss_fn, optimizer)

    #Save embedding layer weights
    torch.save(model.embedding.state_dict(), "trained_embedding_weights.pt")
    
