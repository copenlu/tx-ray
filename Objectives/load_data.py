########################################################################################################################
"""
Description : 1) Loads the data from the data directory.
              2) converts them to token:token_id tensors and id:id_tokens returns it along with the mappings.
Python version : 3.7.3
author Vageesh Saxena
"""
########################################################################################################################


################################################ Importing libraries ###################################################
import os
from io import open

import torch

import nltk

# Custom libraries
from preprocess import DataPreprocessor
########################################################################################################################


class Dictionary(object):
    """
    Creates a dictionary with tokens mapped to their positional ids and vice-versa.
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word.lower() not in self.word2idx:
            self.idx2word.append(word.lower())
            self.word2idx[word.lower()] = len(self.idx2word) - 1
        return self.word2idx[word.lower()]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'wikitext-2-raw-v1', 'wiki.train.raw'))
        self.valid = self.tokenize(os.path.join(path, 'wikitext-2-raw-v1', 'wiki.valid.raw'))
        self.test = self.tokenize(os.path.join(path, 'wikitext-2-raw-v1', 'wiki.test.raw'))

    def tokenize(self, path):
        """
        Takes the train, valid, and test files from the unsup corpus and tokenizes them to form a LongTensors of ids.
        """
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                sent_text = nltk.sent_tokenize(line) # this gives us a list of sentences
                tokenized_text = []
                for sentence in sent_text:
                    current_sent = nltk.word_tokenize(sentence)
                    current_sent.insert(len(current_sent), "eos")
                    tokenized_text.append(current_sent)
                text = [item for sublist in tokenized_text for item in sublist]
                words = text + ['eop']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word.lower())

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                sent_text = nltk.sent_tokenize(line) # this gives us a list of sentences
                tokenized_text = []
                for sentence in sent_text:
                    current_sent = nltk.word_tokenize(sentence)
                    current_sent.insert(len(current_sent), "eos")
                    tokenized_text.append(current_sent)
                text = [item for sublist in tokenized_text for item in sublist]
                words = text + ['eop']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word.lower()]
                    token += 1
        return ids


