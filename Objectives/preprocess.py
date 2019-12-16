########################################################################################################################
"""
Description :  Contains functions related to preprocessing data-
Python version : 3.7.3
author Vageesh Saxena
"""
########################################################################################################################

################################################ Importing libraries ###################################################
import torch

import nltk
from nltk import word_tokenize
########################################################################################################################

class DataPreprocessor:
    """
    Preprocess the data for before aggregrating it in the .tsv files.
    """
    def __init__(self, corpus, id_to_word, word_to_id):
        self.corpus = corpus
        self.id_to_word = id_to_word
        self.word_to_id = word_to_id

    def unsup_model(self):
        """
        :return: para_list(list of paragraphs), w(tokenized list of words), id_to_word, word_to_id
        """
        # Pre-processing data
        # Concatinating test, train, and val data
        unsup_data = torch.cat((self.corpus.train, self.corpus.valid,self.corpus.test), 0)
        # Generating pos tags as per sentence level from nltk library
        # Generates paragraphs on token level again
        para_list = []
        for token in unsup_data:
            para_list.append(self.id_to_word[token])
        # Splitting sentences on the basis of eop
        para_list = " ".join(para_list)
        para_list = para_list.split("eop")
        # Removing empty spaces
        empty_list=[" ",""]
        para_list = [x for x in para_list if x not in empty_list]
        # Tokenizing the sentences
        para_list = [word_tokenize(sent.replace("<","").replace(">","")) for sent in para_list]
        # replacing the keys in old dictionary

        word_to_id = self.word_to_id
        if "<eop>" in word_to_id.keys():
            word_to_id["eop"] = word_to_id["<eop>"]
            del word_to_id['<eop>']

        if "<eos>" in word_to_id.keys():
            word_to_id["eos"] = word_to_id["<eos>"]
            del word_to_id['<eos>']
            
        id_to_word = [w.replace("<eop>","eop").replace("<eos>","eos") for w in self.id_to_word]
        return para_list, id_to_word, word_to_id

    def zero_shot_model(self):
        """
        :return: processed word_to_id and id_to_word mappings.
        """
        word_to_id = self.word_to_id
        if "<eop>" in word_to_id.keys():
            word_to_id["eop"] = word_to_id["<eop>"]
            del word_to_id['<eop>']

        if "<eos>" in word_to_id.keys():
            word_to_id["eos"] = word_to_id["<eos>"]
            del word_to_id['<eos>']

        id_to_word = [w.replace("<eop>","eop").replace("<eos>","eos") for w in self.id_to_word]
        return word_to_id, id_to_word

    def sup_model(self):
        """
        :return: processed dataframe splitted into train, test, and valid for supervised corpus.
        """
        df = self.corpus
        # removing the instances with sentiment neither defined as pos or neg
        df = df[(df["label"] == "neg") | (df["label"] == "pos")]

        # Adding <eos> and <eop> tokens to the raw data in the dataframe
        para_text = []
        for para in df['review']:
            # To tokenize sentences in paragraphs
            sent_text = nltk.sent_tokenize(para)
            tokenized_text = []
            for sentence in sent_text:
                # To tokenize tokens in sentences
                current_sent = nltk.word_tokenize(sentence)
                current_sent.insert(len(current_sent), "eos")
                tokenized_text.append(current_sent)
            text = [item for sublist in tokenized_text for item in sublist]
            text = text + ['eop']
            para_text.append(text)
        
        # Copying the processed data back to the dataframe
        df['review'] = para_text

        # Getting the train, validation, and test data
        train_data = df.loc[df["type"] == "train"]
        test_data = df.loc[df["type"] == "test"]
        # Change the line accordingly. For the sake of showing examples, I have taken 300 paragraphs in total for 
        # consideration.
        val_data = train_data[int(train_data.shape[0]*80/100):]
        train_data = train_data[:int(train_data.shape[0]*80/100)]
        return train_data, test_data, val_data
