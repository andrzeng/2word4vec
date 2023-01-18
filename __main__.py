import vocabulary
import string
import models

if __name__ == '__main__':
    
    '''Initialize array of the words in the Bible, in order'''
    text_bible = []
    with open('bible.txt','r') as f:
        for line in f:
            for word in line.split():
                text_bible.append(word.translate(str.maketrans('','', string.punctuation)).lower())           
