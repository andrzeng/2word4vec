import string
import numpy as np
import random
import torch

'''Initialize array of the words in the Bible, in order'''

text_bible = []
with open('bible.txt','r') as f:
    for line in f:
        for word in line.split():
            text_bible.append(word.translate(str.maketrans('','', string.punctuation)).lower())           
            
def sample_ngram(text, n=5):
    start_index = random.randint(0, len(text)-n)
    n_gram = [text[i] for i in range(start_index, start_index+n)]
    
    return n_gram

def create_embedding(vocab_size, embedding_dim):
    embedding = {}
    for i in range(vocab_size):
        embedding[i] = (torch.rand(embedding_dim)-0.5)*2
    return embedding

def create_lookup_table(text):
    lookup_table = {}
    counter = 0
    for word in text:
        if word not in lookup_table:
            lookup_table[word] = counter
            counter += 1
    return lookup_table