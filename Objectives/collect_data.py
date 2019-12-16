########################################################################################################################
"""
Description :  Collects and aggregrates data for the provided trained models
Python version : 3.7.3
author Vageesh Saxena
"""
########################################################################################################################


################################################ Importing libraries ###################################################
import os
import csv
from io import open

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import flair
from flair.models import SequenceTagger
from flair.data import Sentence
from textblob import TextBlob

import nltk
from nltk import word_tokenize

import pandas as pd

# Custom libraries
from load_data import Corpus
from layers import LSTMLM, LSTMClassifier, LSTMPoolingClassifier
from utilities import tag_sent_tsv_line_unsup, tag_sent_tsv_line_sup, sent_2_index, label_2_index
from preprocess import DataPreprocessor
########################################################################################################################

class DataAggregrator:
    def __init__(self, trained_model, data_dict, corpus, id_to_word, word_to_id, label_to_ix, mode=None):
        self.trained_model = trained_model
        self.data_dict = data_dict
        self.corpus = corpus
        self.id_to_word = id_to_word
        self.word_to_id = word_to_id
        self.label_to_ix = label_to_ix
        self.mode = mode

    def prepare_unsup_data_for_saving(self, para, tagger):
        para.insert(len(para), "eop")
        para = " ".join(para)
        t = TextBlob(para)
        sents = []
        for sent in t.sentences:
            sents.append(Sentence(sent.strip()))
        # Generating POS tags tagged to the sentence
        pos = tag_sent_tsv_line_unsup(sents, tagger, self.data_dict['tagger_type'])
        pos = np.array([item for sublist in pos for item in sublist])
        para = word_tokenize(para)
        para = [torch.tensor(self.word_to_id[w]) if w in self.word_to_id.keys() else torch.tensor(self.word_to_id['unk']) for w in para]
        return para, pos

    def save_unsup_data(self, para, pos, hidden, tsv_writer):
        if len(para) == len(pos):
            for index,token in enumerate(para):
                inputs = token
                encoded_outputs, decoded_outputs, hidden = self.trained_model(inputs.unsqueeze(0).unsqueeze(0), hidden)
                word_weights = torch.nn.functional.softmax(Variable(decoded_outputs.squeeze(0).squeeze(0))).data
                input_word = self.id_to_word[inputs]
                output_word = self.id_to_word[word_weights.argmax()]
                pos_tag = pos[index]
                activation = list(hidden[0].squeeze(0).squeeze(0).data.numpy())
                tsv_writer.writerow([input_word, output_word, activation, pos_tag, np.max(activation), np.argmax(activation)])
                # hidden_activation_dict.setdefault(output_word, []).append(hidden[0].squeeze(0).squeeze(0).data.numpy())
                # Resetting the hidden layer to zero at the end of every sentence.
                if input_word == "eop":
                    hidden = self.trained_model.init_hidden(1)

    def save_sup_data(self, inputs_words, tagger, value_dict, tsv_writer):
        # For nltk.pos_tag to work, we have to use word_tokenize. While using word_tokenizer it is splitting
        # "<" and ">" into individual tokens and thereby its necessary to remove them.
        inputs_sentence = " ".join(inputs_words)
        t = TextBlob(inputs_sentence)
        sents = []
        for sent in t.sentences:
            sents.append(Sentence(sent.strip()))
        pos = tag_sent_tsv_line_sup(sents, tagger, self.data_dict['tagger_type'])    
        value_dict['POS'] = pos

        if len(value_dict['inputs']) == len(value_dict['POS']):
            tsv_writer.writerows(zip(*[value_dict[key] for key in value_dict.keys()]))

    def record_sup_data(self, truth_res, label, para):
        truth_res.append(self.label_to_ix[label])
        para = sent_2_index(para, self.word_to_id)
        label = label_2_index(label, self.label_to_ix)
        y_pred, value_dict = self.trained_model(para, True)
        # The predicted results are processed through BCElosswithlogits,c hence the outputs are
        # passed through the sigmoid layer to turn it into probabilities. 
        y_pred = float(torch.sigmoid(torch.FloatTensor(y_pred)))
        y_pred = [y_pred] * len(value_dict['activations'])
        value_dict["prediction_score"] = y_pred

        actual_label = [int(label)] * len(value_dict['activations'])
        value_dict["labels"] = actual_label

        # Changing the shape and format of data to be written in tsv file
        act_temp = []
        for activation in value_dict["activations"]:
            act = [float(value) for value in activation.squeeze_()]
            act_temp.append(act)
        value_dict['activations'] = act_temp

        inputs_words = []
        for token_id in value_dict["inputs"]:
            inputs_words.append(self.id_to_word[token_id[0]])
        value_dict['inputs'] = np.array(inputs_words)
        return inputs_words, value_dict

    def unsup_data(self):
        """
        :return: Aggregrates unsupervised data for lstm language model in a .tsv file
        """
        # Pre-processing data
        process_data = DataPreprocessor(self.corpus, self.id_to_word, self.word_to_id)
        para_list, id_to_word, word_to_id = process_data.unsup_model()

        # Generating data and saving it in a .tsv file
        self.trained_model.eval()
        hidden = self.trained_model.init_hidden(1)
        tagger = SequenceTagger.load(self.data_dict['tagger_type'])
        
        if self.mode == None:
            filename = self.data_dict["models"]["pretrained_lm"]["saved_tsv_dir"]
        elif self.mode == "1":
            filename = os.path.join("data","unsup_data1.tsv")
        elif self.mode == "48":
            filename = os.path.join("data","unsup_data48.tsv")
        else:
            filename = os.path.join("data","unsup_data49.tsv")

        with open(filename, 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(self.data_dict["models"]["pretrained_lm"]["fields_to_generate"])
            for para in para_list:
                para, pos = self.prepare_unsup_data_for_saving(para, tagger)
                self.save_unsup_data(para, pos, hidden, tsv_writer)


    def sup_data(self):
        """
        :return: Aggregrates supervised data for lstm classifier in a .tsv file
        """
        self.trained_model.eval()
        truth_res, pred_res = ([] for i in range(2))
        tagger = SequenceTagger.load(self.data_dict['tagger_type'])

        with open(self.data_dict["models"]["lstm_classifier"]["saved_tsv_dir"], 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(self.data_dict["models"]["lstm_classifier"]["fields_to_generate"])
            
            for (para, label) in zip(self.corpus["review"],self.corpus["label"]):
                inputs_words, value_dict = self.record_sup_data(truth_res, label, para)
                self.save_sup_data(inputs_words, tagger, value_dict, tsv_writer)

    def sup_pooled_data(self):
        """
        :return: Aggregrates supervised data for lstm classifier in a .tsv file
        """
        self.trained_model.eval()
        truth_res, pred_res = ([] for i in range(2))
        tagger = SequenceTagger.load(self.data_dict['tagger_type'])

        with open(self.data_dict["models"]["lstm_pooled_classifier"]["saved_tsv_dir"], 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(self.data_dict["models"]["lstm_pooled_classifier"]["fields_to_generate"])
            
            for (para, label) in zip(self.corpus["review"],self.corpus["label"]):
                inputs_words, value_dict = self.record_sup_data(truth_res, label, para)
                self.save_sup_data(inputs_words, tagger, value_dict, tsv_writer)

    def save_zero_shot_data(self, pos, para, hidden, tsv_writer):
        if len(pos) == len(para):
            for index,tokens in enumerate(para):
                inputs = tokens
                encoded_outputs, decoded_outputs, hidden = self.trained_model(inputs.unsqueeze(0).unsqueeze(0), hidden)
                word_weights = torch.nn.functional.softmax(Variable(decoded_outputs.squeeze(0).squeeze(0))).data
                input_word = self.id_to_word[inputs]
                output_word = self.id_to_word[word_weights.argmax()]
                pos_tag = pos[index]
                activation = list(hidden[0].squeeze(0).squeeze(0).data.numpy())
                tsv_writer.writerow([input_word, output_word, activation, pos_tag, np.max(activation), np.argmax(activation)])
                # hidden_activation_dict.setdefault(output_word, []).append(hidden[0].squeeze(0).squeeze(0).data.numpy())
                # Resetting the hidden layer to zero at the end of every sentence.
                if input_word == "eos":
                    hidden = self.trained_model.init_hidden(1)

    def prepare_zero_shot_data(self, para, tagger):
        pos = []
        # para = [self.id_to_word[token] for token in para]
        para = " ".join(para)
        t = TextBlob(para)
        sents = []
        for sent in t.sentences:
            sents.append(Sentence(sent.strip()))
        pos = tag_sent_tsv_line_unsup(sents, tagger, self.data_dict['tagger_type'])
        pos = np.array([item for sublist in pos for item in sublist])

        para = word_tokenize(para)
        para = [torch.tensor(self.word_to_id[w]) if w in self.word_to_id.keys() else torch.tensor(self.word_to_id['unk']) for w in para]
        return pos, para

    def zero_shot_data(self):
        """
        :return: Aggregrates zero shot data for unsupervised classifier in a .tsv file
        """
        self.trained_model.eval()
        hidden = self.trained_model.init_hidden(1)
        tagger = SequenceTagger.load(self.data_dict['tagger_type'])

        with open(self.data_dict["models"]["zero_shot"]["saved_tsv_dir"], 'w') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(self.data_dict["models"]["zero_shot"]["fields_to_generate"])

            for para in list(self.corpus['review']):
                # converting paragraphs into tokens
                pos, para = self.prepare_zero_shot_data(para, tagger)
                self.save_zero_shot_data(pos, para, hidden, tsv_writer)
            


