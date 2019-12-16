########################################################################################################################
"""
Description :  Contains the utility functions for the module.
Python version : 3.7.3
author Vageesh Saxena
"""
########################################################################################################################

################################################ Importing libraries ###################################################
import re

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

import torch.optim as optim

import matplotlib.pyplot as plt
########################################################################################################################

def plot_loss_and_perplexity_for_language_model(data_list1, data_list2, epoch_list, figure_size=(7,1), dpi_value=300, figure_name=''):
    """
    :param data_list1: loss list(dtype: list)
    :param data_list2: perplexity list(dtype: list)
    :param epoch_list: epoch list(dtype: list)
    :param figure_name: Name of the figure to be saved(dtype: string)
    :return: Plots the Loss and perplexity of the language model.
    """
    fig1 = plt.figure(figsize=figure_size, dpi=dpi_value)
    plt.plot(epoch_list, data_list1, 'b', label='val_loss')
    plt.plot(epoch_list, data_list2, 'r', label='perplexity')
    plt.xlabel("Epochs")
    plt.ylabel("Loss and Perplexity")
    plt.title("Loss-Perplexity curve for " + figure_name + " data" )
    fig1.savefig(figure_name + "_loss_curve.png", bbox_inches='tight')


def batchify(data, bsz, device):
    """
    :param data: data corpus(could be train, test, or validation dataset)
    :param bsz: Batch size(dtype: int32)
    :param device: GPU/CPU(dtype: torch.device)
    :return: dataset divided into batches
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """
    :param h: hidden state(dtype: torch.tensor)
    :return: Wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def make_batches(data, bptt, i):
    """
    :param data: data corpus(could be train, test, or validation dataset)
    :param bptt: Backpropogation through time or sequence length(dtype: int32)
    :param i: Iterated chunks(dtype: int32)
    :return: subdivides the input data into chunks of length bptt and generates source and targets for model to train
    """
    seq_len = min(bptt, len(data) - 1 - i)
    inputs = data[i:i+seq_len]
    targets = data[i+1:i+1+seq_len].view(-1)
    return inputs, targets


def tag_sent_tsv_line_sup(sents, tagger, tag_type):            
    """
    :param sents: sentences in paragraphs(dtype:list of strings)
    :param tagger: POS/NER tagger from flair/nltk
    :param tag_type: tag type('pos'/'ner')
    :return: array of tagged sentences with their associated 'pos'/'ner' for supervised corpus
    """
    tagger.predict(sents)
    tags = []
    for s in sents:                                        # tag a batch of sentence and pipe out tsv lines
        temp_tags = [str(t.get_tag(tag_type)) for t in s]       # throws error for wrong tag type
        tags.append([re.sub(r'\([^)]*\)', '', tag) for tag in temp_tags])
    return tags[0]

def tag_sent_tsv_line_unsup(sents, tagger, tag_type):            
    """
    :param sents: sentences in paragraphs(dtype:list of strings)
    :param tagger: POS/NER tagger from flair/nltk
    :param tag_type: tag type('pos'/'ner')
    :return: array of tagged sentences with their associated 'pos'/'ner' for unsupervised corpus
    """
    tagger.predict(sents)
    tags = []
    for s in sents:                                        # tag a batch of sentence and pipe out tsv lines
        temp_tags = [str(t.get_tag(tag_type)) for t in s]       # throws error for wrong tag type
        tags.append([re.sub(r'\([^)]*\)', '', tag) for tag in temp_tags])
    return tags


def sent_2_index(seq, to_ix, cuda=False):
    """
    :param seq: sequence list with pararaphs denoted by list of tokens(dtype:list).
    :param to_ix: word to index mappings(dtype:dict)
    :return: Long tensor for all the tokens converted to their respective ids
    """
    var = autograd.Variable(torch.LongTensor([to_ix[w.lower()] if w.lower() in to_ix.keys() else to_ix["unk"] for w in seq]))
    return var

def label_2_index(label, label_to_ix, cuda=False):
    """
    :param label: sequence list of labels(dtype:list).
    :param label_to_ix: labels to index mappings(dtype:dict)
    :return: Long tensor for all the labels converted to their respective ids(negative being zero and positive being one)
    """ 
    var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
    return var

def evaluate_supervised_model(model, data, loss_function, word_to_ix, label_to_ix, data_acc_list, data_roc_list,
                             data_loss_list, name ='valid'):
    """
    :param model: trained model
    :param data: data to evaluated on(dtype: pandas.core.frame.DataFrame)
    :param loss_function: loss function used for evaluation
    :param word_to_ix: word to index mappings(dtype: dict)
    :param label_to_ix: label to index mappings(dtype: dict)
    :param data_acc_list: a list to collect accuracy at every epoch(dtype: list)
    :param data_roc_list: a list to collect roc score at every epoch(dtype: list)
    :param data_loss_list: a list to collect loss at every epoch(dtype: list)
    :param name: type of data(Could be 'train','test',or 'valid'. dtype: string)
    :return: evaluated accuracy and roc on the entire dataset, data_acc_list, data_roc_list, data_loss_list
    """
    model.eval()
    avg_loss = 0.0
    truth_res, pred_res = ([] for i in range(2))
    
    with torch.no_grad():
        for (para, label) in zip(data["review"],data["label"]):
            truth_res.append(label_to_ix[label])
            para = sent_2_index(para, word_to_ix)
            label = label_2_index(label, label_to_ix)
            y_pred, value_dict = model(para, True)
            # The predicted results are processed through BCElosswithlogits, hence the outputs are
            # passed through the sigmoid layer to turn it into probabilities. 
            pred_res.append(float(torch.sigmoid(torch.FloatTensor(y_pred))))
            # Since the loss_function already has a sigmoid layer attached to it, we don't need to pass the predictions
            # again through another sigmoid layer.
            loss = loss_function(y_pred, label.type(torch.FloatTensor))
            avg_loss += loss.item()
    
    avg_loss /= len(data)
    roc = roc_auc_score(np.array(truth_res), np.array(pred_res).round(), sample_weight=None)
    pred_res = [0 if values > 0.5 else 1 for values in pred_res]
    acc = accuracy_score(np.array(truth_res), np.array(pred_res).round(), sample_weight=None)
    
    data_roc_list.append(roc)
    data_loss_list.append(avg_loss)
    data_acc_list.append(acc)
    
    print(' '*16 + name + ':|avg_loss:%g|ROC:%g|Accuracy:%g|' % (avg_loss, roc, acc))
    return acc, roc, data_acc_list, data_roc_list, data_loss_list

def train_supervised_model(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i, train_acc_list, 
                           train_roc_list, train_loss_list, batch_size=32, clip=5):
    """
    :param model: the model to be trained
    :param train_data: Training data(dtype: pandas.core.frame.DataFrame)
    :param loss_function: loss function used for evaluation
    :param optimizer: Optimizer used while training
    :param word_to_ix: word to index mappings(dtype: dict)
    :param label_to_ix: label to index mappings(dtype: dict)
    :param i: number of steps passed(dtype: int)
    :param train_acc_list: a list to collect accuracy at every epoch(dtype: list)
    :param train_roc_list: a list to collect roc score at every epoch(dtype: list)
    :param train_loss_list: a list to collect loss at every epoch(dtype: list)
    :param batch_size: batch size(dtype: int)
    :param clip: clip rate(dtype: int)
    :return: train_acc_list, train_roc_list, train_loss_list
    """
    model.train()
    truth_res, pred_res = ([] for i in range(2))
    avg_loss = 0.0
    count = 0
    
    for (para, label) in zip(train_data["review"],train_data["label"]):
        truth_res.append(label_to_ix[label])        
        para = sent_2_index(para, word_to_ix)
        label = label_2_index(label, label_to_ix)
        y_pred, value_dict = model(para)
        # The predicted results are processed through BCElosswithlogits, hence the outputs are
        # passed through the sigmoid layer to turn it into probabilities. 
        pred_res.append(float(torch.sigmoid(torch.FloatTensor(y_pred))))
        # Since the loss_function already has a sigmoid layer attached to it, we don't need to pass the predictions
        # again through another sigmoid layer.
        loss = loss_function(y_pred, label.type(torch.FloatTensor))
        loss.backward()
        current_loss = loss.item()
        avg_loss += current_loss
        
        count += 1
        if count % 10000 == 0:
            print('|paragraphs: %d|loss :%g|' % (count, current_loss))
        
        if count % batch_size == 0:
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            #raise Exception("Arrange clip as per your batch size")
            #raise Exception("Try with and without clipping")
            if clip != None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()
            optimizer.zero_grad()
            
            
    avg_loss /= len(train_data)
    print('-' * 100)
    
    train_loss_list.append(avg_loss)
    roc = roc_auc_score(np.array(truth_res), np.array(pred_res).round(), sample_weight=None)
    pred_res = [0 if values > 0.5 else 1 for values in pred_res]
    acc = accuracy_score(np.array(truth_res), np.array(pred_res).round(), sample_weight=None)
    
    train_roc_list.append(roc)
    train_acc_list.append(acc)
    
    print('|End of Epoch:%d|Training data:|avg_loss:%g|ROC:%g|Accuracy:%g|'%(int(i+1), avg_loss, roc, acc))
    return train_acc_list, train_roc_list, train_loss_list

def train_model_and_generate_plots(model, train_data, test_data, val_data, unsup_word_to_idx, label_to_idx, learning_rate, batch_size, 
                                   nb_epochs, save_dir, description, early_stopping=5):
    """
    :param model: the model to be trained
    :param train_data: Training data(dtype: pandas.core.frame.DataFrame)
    :param test_data: Test data(dtype: pandas.core.frame.DataFrame)
    :param val_data: Validation data(dtype: pandas.core.frame.DataFrame)
    :param unsup_word_to_idx: word to index mappings(dtype: dict)
    :param label_to_idx: label to index mappings(dtype: dict)
    :param learning_rate: Learning rate(dtype:float)
    :param nb_epochs: number of Epochs(dtype:int)
    :param save_dir: directory for the model to be saved(dtype:string)
    :param batch_size: Batch size(dtype:int)
    :param early_stopping: After how many steps should the model stop training if the val_roc doesn't change(dtype:int)
    :param description: Data desciption(Train,test, or validation; dtype:string)
    """
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    train_acc_list, train_roc_list, train_loss_list, test_acc_list, test_roc_list, test_loss_list, val_acc_list, val_roc_list, val_loss_list, epoch_list = ([] for i in range(10))
    
    #optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
    no_up = 0
    best_val_roc = 0.0
    
    for i in range(nb_epochs):
        epoch_list.append(int(i+1))
        print('epoch: %d start!' % int(i+1))
        
        # Training the model
        optimizer.zero_grad()
        # pbar = tqdm(total=train_data.shape[0])
        train_acc_list, train_roc_list, train_loss_list = train_supervised_model(model, train_data, loss_function, optimizer, unsup_word_to_idx, label_to_idx, i, train_acc_list, train_roc_list, train_loss_list, batch_size)
        
        # Hyper-tuning the model
        optimizer.zero_grad()
        val_acc, val_roc, val_acc_list, val_roc_list, val_loss_list = evaluate_supervised_model(model, val_data, loss_function, unsup_word_to_idx, label_to_idx, val_acc_list, val_roc_list, val_loss_list, 'validation data')
        
        # Testing the model
        optimizer.zero_grad()
        test_acc, test_roc, test_acc_list, test_roc_list, test_loss_list = evaluate_supervised_model(model,test_data, loss_function, unsup_word_to_idx, label_to_idx, test_acc_list, test_roc_list, test_loss_list, 'test data')
        
        # Un-comment the below lines if you want to save models with smallest change in val_acc and val_roc
        """if (val_acc > best_val_acc) and (val_roc <= best_val_roc):
            # Saving models on the basis of validation accuracy
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_models_lstm_1500/acc_' + str(int(test_acc*100))
                       + "_roc_" + str(int(test_roc*100)) + '.pt')
            no_up = 0
        elif (val_roc > best_val_roc) and (val_acc <= best_val_acc):
            # Saving models on the basis of validation roc
            best_val_roc = val_roc
            torch.save(model.state_dict(), 'best_models_lstm_1500/roc_' + str(int(test_roc*100))
                       + "_acc_" + str(int(test_acc*100)) + '.pt')
            no_up = 0
        elif (val_roc > best_val_roc) and (val_acc > best_val_acc):
            # Saving models on the basis of validation roc and validation accuracy
            best_val_roc = val_roc
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_models_lstm_1500/combined_roc_' + str(int(test_roc*100))
                       + "_acc_" + str(int(test_acc*100)) + '.pt')
            no_up = 0"""

        if val_roc > best_val_roc:
            torch.save(model.state_dict(), save_dir)
            best_val_acc = val_roc

        else:
            # early stopping
            no_up += 1
            if no_up >= 5:
                break
    
    # Un-comment the below lines to generate training, test, and validation plots
    """
    # Saving the lists in a dataframe so that it can be used to plot the variations wrt epochs.
    df = pd.DataFrame({"epochs":epoch_list, "train_acc": train_acc_list, "train_roc": train_roc_list,
                       "train_loss":train_loss_list, "val_acc" : val_acc_list, "val_roc": val_roc_list, "val_loss" :
                       val_loss_list, "test_acc" : test_acc_list, "test_roc": test_roc_list, "test_loss" : test_loss_list})

    plot = df.plot(x="epochs",y=["train_acc","test_acc","val_acc"],title= "Accuracy curve")
    fig = plot.get_figure()
    fig.savefig(description + "_acc.png")
        
    plot = df.plot(x="epochs",y=["train_loss","test_loss","val_loss"],title="Loss curve")
    fig = plot.get_figure()
    fig.savefig(description + "_loss.png")
    
    plot = df.plot(x="epochs",y=["train_roc","test_roc","val_roc"],title="ROC curve")
    fig = plot.get_figure()
    fig.savefig(description + "_roc.png")
    """

    return model
