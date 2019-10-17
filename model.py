# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:05:53 2019

@author: tungo
"""

import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        embedding = self.embedding(x)
        out, hidden = self.gru(embedding)
        last_backward_hidden = out[:, -1, self.hidden_size:].unsqueeze(0)
        last_forward_hidden = hidden[0].unsqueeze(0)
        return out, last_backward_hidden, last_forward_hidden
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_dim):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.Wa = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ua = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.Va = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
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