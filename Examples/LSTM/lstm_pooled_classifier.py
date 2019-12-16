########################################################################################################################
"""
Description :  Takes the trained unsupervised laguage model trained on Wikitext-2 and apply supervision by performing
               classification with pooling on the IMDB movie review dataset.
Python version : 3.5.3
author Vageesh Saxena
"""
########################################################################################################################


############################################ Loading Libraries ########################################################
# General libraries
import os
import argparse
import re
import pickle
import time
import sys
import csv
import random
from tqdm import tqdm
from collections import defaultdict

import nltk
from nltk import pos_tag, word_tokenize
nltk.download("punkt")

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

import pandas as pd
import numpy as np

#custom library
sys.path.append('../../Objectives/')
from load_data import Corpus
from preprocess import DataPreprocessor
from layers import LSTMLM, LSTMPoolingClassifier
from utilities import train_model_and_generate_plots

import warnings
warnings.filterwarnings('ignore')
##########################################################################################################################


############################################## Setting up the parser ####################################################
parser = argparse.ArgumentParser(description='LSTM based Classifier Model')

# Model parameters.
parser.add_argument('--data_dir', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--pretrained_lm', type=str, default='./trained_models/LSTM/trained_lm.pt', help='Pretrained Language model directory')
parser.add_argument('--emsize', type=int, default=100, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=100, help='number of hidden units per layer')
parser.add_argument('--dropout', type=float, default=0.65, help="recurrent dropout for the trained sequence encoder model")
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--nb_epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--save', type=str, default='trained_models/LSTM/pooled_classifier.pt', help='directory for the model to be saved')

args = parser.parse_args()
########################################################################################################################


# Loading the unsupervised data
unsup_corpus = Corpus(args.data_dir)
unsup_word_to_idx = unsup_corpus.dictionary.word2idx
unsup_idx_to_word = unsup_corpus.dictionary.idx2word
ntokens = len(unsup_corpus.dictionary)

# Loading the pretrained model
device = torch.device('cuda')
pre_trained_lm = LSTMLM(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout)
pre_trained_lm.load_state_dict(torch.load(args.pretrained_lm, map_location=device), strict=True)

# Loading the supervised dataset
df = pd.read_csv(os.path.join(args.data_dir,"imdb_master.csv"), encoding="cp1252")
df = df.sample(frac=1).reset_index(drop=True).iloc[:300]
# Deleting unnecessary columns and Pre-processing data
del df['Unnamed: 0']
del df['file']
process_data = DataPreprocessor(df, unsup_idx_to_word, unsup_word_to_idx)
train_df, test_df, valid_df = process_data.sup_model()

imdb_label_to_idx = {"neg":0,"pos":1}
print('vocab size:',len(unsup_word_to_idx),'label size:',len(imdb_label_to_idx))

# Loading the supervised model
sup_model = LSTMPoolingClassifier(embedding_dim=args.emsize,hidden_dim=args.nhid, vocab_size=len(unsup_word_to_idx), 
                       label_size=len(imdb_label_to_idx), pre_trained_model = pre_trained_lm, 
                       wiki_idx_to_word=unsup_idx_to_word)
sup_model = train_model_and_generate_plots(sup_model, train_df, test_df, valid_df, unsup_word_to_idx, imdb_label_to_idx, 
                                           args.lr, args.batch_size, args.nb_epochs, args.save, None)

