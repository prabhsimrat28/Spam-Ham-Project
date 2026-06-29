import string
from nltk.tokenize import word_tokenize
import pickle
import pandas as pd

with open("models/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

exclude=string.punctuation


def remove_punc(text):
    return text.translate(str.maketrans('','',exclude))


def tokeniser(text):
    return word_tokenize(text)



def sentence2int(text):
    encoded=[]
    for word in text:
        if word in vocab:
            encoded.append(vocab[word])
        else:
            encoded.append(1)
    return encoded


def preprocess_data(data):
   data=data.lower()
   data=remove_punc(data)
   data=tokeniser(data)
   data=sentence2int(data)
   return data
   

