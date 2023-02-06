import string
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



def get_ngram(text, idx,  n=5):
    start_index = idx 
    n_gram = [text[i] for i in range(start_index, start_index+n)]
    #n_gram = np.array(n_gram)
    return n_gram

def sample_ngram(text, n=5):
    start_index = random.randint(0, len(text)-n)
    n_gram = [text[i] for i in range(start_index, start_index+n)]
    #n_gram = np.array(n_gram)
    return n_gram

def create_lookup_table(text):
    lookup_table = {}
    counter = 0
    for word in text:
        if word not in lookup_table:
            lookup_table[word] = counter
            counter += 1
    return lookup_table

class CorpusDataset(Dataset):
    def __init__(self, text_file: str, ngram_len: int, lookup_table: dict):
        '''Initialize array of the words in the Bible, in order'''
        text = []
        with open(text_file,'r') as f:
            for line in f:
                for word in line.split():
                    #text.append(word.translate(str.maketrans('','', string.punctuation)).lower())           
                    text.append(lookup_table[word.translate(str.maketrans('','', string.punctuation)).lower()])           

        #Move sliding window over the text and collect all n-grams
        self.all_ngrams = []
        for idx in range(len(text) - ngram_len):
            self.all_ngrams.append(get_ngram(text, idx, n=ngram_len))

        self.obscured_phrases = []
        self.target_words = []
        for phrase in self.all_ngrams:
          #  print( phrase[int((ngram_len+1)/2):])
            self.obscured_phrases.append(torch.LongTensor(phrase[:int((ngram_len-1)/2)] + phrase[int((ngram_len+1)/2):]))
            self.target_words.append(phrase[int((ngram_len-1)/2)])

    def __len__(self):
        return len(self.all_ngrams)

    def __getitem__(self, idx):
        return self.obscured_phrases[idx], self.target_words[idx]

    