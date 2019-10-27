# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:05:53 2019

@author: tungo
"""

import torch.nn as nn
import torch

class Encoder(nn.Module):
    """
    Encoder using bi-directional GRU to encode input sentences
    
    Arguments:
    - vocab_size, embedding_dim, hidden_size: integers
    - modified: False to use Encoder's last backward hidden state like in original paper. True to use a combination between forward and backward hidden states
    
    Inputs:
    - x: batch of input sentences after converting to indices, size (batch, Tx)
    
    Returns:
    - out: output from GRU
    - last_backward_hidden, last_forward_hidden: hidden states of backward and forward GRU at the last word of input sentences
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, modified=False):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.modified = modified
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, batch_first=True, bidirectional=True)
        if self.modified:
            self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x):
        embedding = self.embedding(x)
        out, hidden = self.gru(embedding)
        last_backward_hidden = out[:, 0, self.hidden_size:].unsqueeze(0)
        last_forward_hidden = hidden[0].unsqueeze(0)
        if self.modified:
            enc_hidden = self.fc_hidden(torch.cat((last_backward_hidden, last_forward_hidden), dim=-1))
        else:
            enc_hidden = last_backward_hidden
        return out, enc_hidden
    
class Decoder(nn.Module):
    """
    Decoder with Attention for 1 timestep.
    
    Arguments:
    - hidden_size, vocab_size, embedding_dim: integers
    
    Inputs:
    - dec_input: current input
    - hidden: hidden state from previous timestep
    - enc_out: output from Encoder
    
    Returns:
    - out: Decoder's output
    - hidden: hidden states to feed into next timestep's Decoder
    """
    def __init__(self, hidden_size, vocab_size, embedding_dim):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Alignment model
        self.Wa = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ua = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.Va = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        
        # GRU layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim + self.hidden_size * 2, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, dec_input, hidden, enc_out):
        Tx = enc_out.shape[1]
        hidden_repeat = hidden.permute(1, 0, 2).repeat(1, Tx, 1)
        energies = self.Va(torch.tanh(self.Wa(hidden_repeat) + self.Ua(enc_out)))
        alphas = self.softmax(energies)
        context = torch.sum(alphas * enc_out, dim=1).unsqueeze(1)
        embedding = self.embedding(dec_input.unsqueeze(1))
        gru_input = torch.cat((embedding, context), dim=-1)
        out, hidden = self.gru(gru_input, hidden.contiguous())
        out = self.out(out)
        return out, hidden