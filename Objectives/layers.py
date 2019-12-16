########################################################################################################################
"""
Description :  Consists architectural classes for the Unuspervised language and Supervised classifier models.
Python version : 3.7.3
author Vageesh Saxena
"""
########################################################################################################################


################################################ Importing libraries ###################################################
import torch
import torch.nn as nn
########################################################################################################################


class LSTMLM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(LSTMLM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.lstm = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.lstm = nn.LSTM(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Enabling weight tying
        # Whenever the dimensions of inputs and ouputs are same, weights can be shared in between to reduce the number
        # of training parameters and improve the performance.
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        # Uniformly initializing weights between two values
        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        encoded_output_from_embedding_layer, hidden = self.lstm(emb, hidden)
        output = self.drop(encoded_output_from_embedding_layer)
        decoded_output_from_linear_layer = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return encoded_output_from_embedding_layer, decoded_output_from_linear_layer.view(output.size(0), output.size(1), decoded_output_from_linear_layer.size(1)), hidden
    
    def init_hidden(self, batch_size):
        # Initializes the hidden and cell state
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                    weight.new_zeros(self.nlayers, batch_size, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, batch_size, self.nhid)


class LSTMClassifier(nn.Module):
    """
    1) mode for the model can either be train or generate
    2) When the model is in training mode, it takes the entire paragraph and pass it through nn.LSTM to make the 
       training efficient.
    3) When the model is in generation mode, it takes one token at time from the paragraphs and update the hidden
       activation to get intermediate data.
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, pre_trained_model, wiki_idx_to_word, activations_to_prune=None):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        # The label size is been set to one for IMDB movie review dataset so to use BCEWithLogitsLoss.
        self.linear_layer = nn.Linear(hidden_dim, 1)
        self.pretrained_model = pre_trained_model
        self.wiki_idx_to_word = wiki_idx_to_word

        # Neurons to prune
        self.prunned_activations = activations_to_prune
        
    def init_hidden(self):
        return (torch.tensor(torch.zeros(1, 1, self.hidden_dim)),
                torch.tensor(torch.zeros(1, 1, self.hidden_dim)))
    
    def forward(self, paragraphs, eval_flag=False):
        # Getting  modified hidden state from the pretrained model  
        if eval_flag:
            # To disable batchnorm and dropout layers in evaluation and test mode
            self.pretrained_model.eval()
        else:
            # To stop the gradient flow back during back-propagation
            self.pretrained_model.train()
        
        # Freezing the embedding layer of the pretrained model
        self.pretrained_model.encoder.weight.requires_grad = False
        value_dict = {}
        
        # The hidden state of the lSTM unit is supposed to be  (num_layers * num_directions, batch, hidden_size)
        hidden = self.init_hidden()
        # The input to the LSTM unit is supposed to be (seq_len, batch, input_size)
        # Here the seq_length is set to the length of paragraphs
        # batch_size is set to one.
        paragraphs = paragraphs.resize_((paragraphs.size(0),1))
        # Getting the input hidden activations
        encoded_outputs, decoded_outputs, hidden = self.pretrained_model(paragraphs, hidden)
        # Getting the output hidden activation(i.e the hidden state is a tuple with (input_hidden,output_hidden))
        # hidden[0] indicates the output hidden activation and hidden[1] indicates insput hidden activation
        hidden_out = hidden[0]
        
        if self.prunned_activations is not None:
            # Performing pruning
            hidden_out_temp = hidden[0].view(-1)
            hidden_in_temp = hidden[1].view(-1)

            for activation in self.prunned_activations:
                hidden_out_temp[activation] = 0.0000
                # hidden_in_temp[activation] = 0.0000
            hidden = (hidden_out_temp.view(1,1,1500),hidden_in_temp.view(1,1,1500))
        
        value_dict["inputs"] = paragraphs.data.numpy()
        activations = encoded_outputs.detach()
        max_activations = [float(act.squeeze_().max()) for act in activations]
        max_activation_index = [int(act.squeeze_().argmax()) for act in activations]
                
        value_dict["activations"] = activations
        value_dict["max_activations"] = max_activations
        value_dict["max_activation_index"] = max_activation_index

        # hidden state is a 3D tensor of (seq_length, batch_size, hidden_activation) since we are only concerned
        # with the last hidden activation we take [-1] index and send it through the linear layer
        y_logits  = self.linear_layer(hidden_out[0][-1])
        # BCEwithlogitsloss already has a sigmoid layer in it and the line below is for BCEwithlogitsloss.
        return y_logits, value_dict


class LSTMPoolingClassifier(nn.Module):
    """
    1) mode for the model can either be train or generate
    2) When the model is in training mode, it takes the entire paragraph and pass it through nn.LSTM to make the 
       training efficient.
    3) When the model is in generation mode, it takes one token at time from the paragraphs and update the hidden
       activation to get intermediate data.
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, pre_trained_model, wiki_idx_to_word):
        super(LSTMPoolingClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        # The label size is been set to one for IMDB movie review dataset so to use BCEWithLogitsLoss.
        self.linear_layer = nn.Linear(3*hidden_dim, 1)
        self.pretrained_model = pre_trained_model
        self.wiki_idx_to_word = wiki_idx_to_word
            
    def init_hidden(self):
        return (torch.tensor(torch.zeros(1, 1, self.hidden_dim)),
                torch.tensor(torch.zeros(1, 1, self.hidden_dim)))
    
    def forward(self, paragraphs, eval_flag=False):
        # Getting  modified hidden state from the pretrained model  
        if eval_flag:
            # To disable batchnorm and dropout layers in evaluation and test mode
            self.pretrained_model.eval()
        else:
            # To stop the gradient flow back during back-propagation
            self.pretrained_model.train()
        
        # Freezing the embedding layer of the pretrained model
        self.pretrained_model.encoder.weight.requires_grad = False
        value_dict = {}
        
        # The hidden state of the lSTM unit is supposed to be  (num_layers * num_directions, batch, hidden_size)
        hidden = self.init_hidden()
        # The input to the LSTM unit is supposed to be (seq_len, batch, input_size)
        # Here the seq_length is set to the length of paragraphs
        # batch_size is set to one.
        paragraphs = paragraphs.resize_((paragraphs.size(0),1))
        # Getting the input hidden activations
        encoded_outputs, decoded_outputs, hidden = self.pretrained_model(paragraphs, hidden)
        # Getting the output hidden activation(i.e the hidden state is a tuple with (input_hidden,output_hidden))
        # hidden[0] indicates the output hidden activation and hidden[1] indicates insput hidden activation
        hidden_out = hidden[0]
        # bringing hidden_out in the shape of (1,1500) from (1,1,1500)
        hidden_out = hidden_out.squeeze(1)

        average_hidden = torch.nn.functional.adaptive_avg_pool1d(encoded_outputs.permute(1,2,0), (1,)).view(1,-1)
        max_hidden = torch.nn.functional.adaptive_max_pool1d(encoded_outputs.permute(1,2,0), (1,)).view(1,-1)
        # Concatenating hidden activations with avg pooling and max pooling
        cat_hidden = torch.cat([hidden_out,average_hidden,max_hidden], 1)
        
        value_dict["inputs"] = paragraphs.data.numpy()
        activations = encoded_outputs.detach()
        max_activations = [float(act.squeeze_().max()) for act in activations]
        max_activation_index = [int(act.squeeze_().argmax()) for act in activations]
                
        value_dict["activations"] = activations
        value_dict["max_activations"] = max_activations
        value_dict["max_activation_index"] = max_activation_index

        # hidden state is a 3D tensor of (seq_length, batch_size, hidden_activation) since we are only concerned
        # with the last hidden activation we take [-1] index and send it through the linear layer
        y_logits  = self.linear_layer(cat_hidden[-1])
        # BCEwithlogitsloss already has a sigmoid layer in it and the line below is for BCEwithlogitsloss.
        return y_logits, value_dict
