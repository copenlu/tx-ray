########################################################################################################################
"""
Description :  Trains and evaluates the LSTM language model on unsupervised(Wikitext-2) data.
Python version : 3.5.3
author Vageesh Saxena
"""
########################################################################################################################


################################################ Importing libraries ###################################################
import argparse
import time
import math
import os
import sys

import torch
import torch.nn as nn

#custom library
sys.path.append('../../Objectives/')
from load_data import Corpus
from layers import LSTMLM
from utilities import plot_loss_and_perplexity_for_language_model, batchify, repackage_hidden, make_batches
########################################################################################################################


############################################## Setting up the parser ####################################################
parser = argparse.ArgumentParser(description='LSTM based Language Model')

# Model parameters.
parser.add_argument('--data_dir', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=50, help='sequence length')
parser.add_argument('--fig_name', type=str, default='Validation', help='Name of the figure to save loss-perplexity curve')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--save', type=str, default='trained_lm.pt', help='path to save the final model')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')

args = parser.parse_args()
########################################################################################################################


######################################### Initializing Global parameters ###############################################
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")
########################################################################################################################


############################################### Loading data ###########################################################
eval_batch_size = args.batch_size // 2
corpus = Corpus(args.data_dir)
train_data = batchify(corpus.train, args.batch_size, device)
val_data = batchify(corpus.valid, eval_batch_size, device)
test_data = batchify(corpus.test, eval_batch_size, device)
ntokens = len(corpus.dictionary)
########################################################################################################################


############################################## Training the model ######################################################
# Building the model on GPU
model = LSTMLM(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
criterion = nn.CrossEntropyLoss()

train_loss_list, train_perplexity_list, train_epoch_list = ([] for i in range(3))
val_loss_list, val_perplexity_list, val_epoch_list = ([] for i in range(3))
test_loss_list, test_perplexity_list, test_epoch_list = ([] for i in range(3))
best_val_loss = None
lr = args.lr

# Starting to iterate over all the epochs
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()

    # Turning on training mode (enabling the dropout)
    model.train()

    total_loss = 0.0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        inputs, targets = make_batches(train_data, args.bptt, i)
        # Starting each batch, we detach the hidden state and cell state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        encoded_output_from_embedding_layer, decoded_output_from_linear_layer, hidden  = model(inputs, hidden)
        loss = criterion(decoded_output_from_linear_layer.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'train_loss {:5.2f} | train_ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            
            # Recording training data
            train_loss_list.append(cur_loss)
            train_epoch_list.append(epoch)
            train_perplexity_list.append(math.exp(cur_loss))

            total_loss = 0
            start_time = time.time()
    
    # evaluating on the validation data
    # Turn on evaluation mode will disables dropout.
    model.eval()

    total_loss = 0.0
    hidden = model.init_hidden(eval_batch_size)
    
    with torch.no_grad():
        for i in range(0, val_data.size(0) - 1, args.bptt):
            inputs, targets = make_batches(val_data, args.bptt, i)
            encoded_outputs, decoded_output, hidden = model(inputs, hidden)
            output_flat = decoded_output.view(-1, ntokens)
            total_loss += len(inputs) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)

    val_loss = total_loss / (len(val_data) - 1)
    
    val_loss_list.append(val_loss)
    val_perplexity_list.append(math.exp(val_loss))
    val_epoch_list.append(epoch)

    print('-' * 100)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 100)

    if not os.path.exists(os.path.join("trained_models", "LSTM")):
        os.makedirs(os.path.join("trained_models", "LSTM"))

    # Saving the model with lowest validation loss
    if not best_val_loss or val_loss < best_val_loss:
        # model_name = "epoch_" + str(epoch) + "_val_loss_" + str(val_loss) + "_ppl_" + str(math.exp(val_loss)) + ".pt"
        with open(os.path.join("trained_models", "LSTM", args.save), 'wb') as f:
            torch.save(model.state_dict(), f)
        best_val_loss = val_loss
    else:
        # reduce the learning rate
        lr /= 4.0
    best_val_loss = val_loss
    
    # Saving model every 10 epochs for test data
    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        total_loss = 0.0
        hidden = model.init_hidden(eval_batch_size)
    
        with torch.no_grad():
            for i in range(0, test_data.size(0) - 1, args.bptt):
                inputs, targets = make_batches(test_data, args.bptt, i)
                encoded_output_from_embedding_layer, decoded_output_from_linear_layer, hidden = model(inputs, hidden)
                output_flat = decoded_output_from_linear_layer.view(-1, ntokens)
                total_loss += len(inputs) * criterion(output_flat, targets).item()
                hidden = repackage_hidden(hidden)

        test_loss = total_loss / (len(test_data) - 1)
        model_name = "epoch_" + str(epoch) + "_test_loss_" + str(test_loss) + "_ppl_" + str(math.exp(test_loss)) + ".pt"
        test_loss_list.append(test_loss)
        test_perplexity_list.append(math.exp(test_loss))
        test_epoch_list.append(epoch)
        
        print('-' * 100)
        print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
          'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     test_loss, math.exp(test_loss)))
        print('-' * 100)

        with open(os.path.join("trained_models", "LSTM", model_name), 'wb') as f:
            torch.save(model.state_dict(), f)    
######################################################################################################################


# loss and perplexities plots per epochs for train, test, and validation data  
plot_loss_and_perplexity_for_language_model(train_loss_list, train_perplexity_list, train_epoch_list, figure_name='train')
plot_loss_and_perplexity_for_language_model(val_loss_list, val_perplexity_list, val_epoch_list, figure_name='validation')
plot_loss_and_perplexity_for_language_model(test_loss_list, test_perplexity_list, test_epoch_list, figure_name='test')
