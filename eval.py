# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:09:25 2019

@author: tungo
"""
import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

def validate(loader, encoder, decoder, id2word, device='cpu'):
    """
    Validate model's performance on validation set using BLEU-4 score.
    
    Inputs:
    - val_loader: DataLoader for validation set
    - encoder: an Encoder model
    - decoder: a Decoder model
    - id2word: id2word from target training set
    - device: 'cpu' or 'cuda'
    
    Return:
    - bleu: BLEU-4 score of dataset
    """
    encoder.eval()
    decoder.eval()
    references, hypotheses = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=torch.long)
            y = y.to(device=device, dtype=torch.long)
            enc_out, enc_hidden = encoder(x)
            dec_hidden = enc_hidden
            dec_input = y[:, 0]
            ref_matrix = y.clone().cpu().numpy()
            for vec in ref_matrix:
                sentence = [id2word[id] for id in vec[1:] if id2word[id] not in ['<END>', '<PAD>']]
                references.append([sentence])
            hypo_matrix = []
            for t in range(1, y.size(1)):
                out, dec_hidden = decoder(dec_input, dec_hidden, enc_out)
                top = torch.max(out, dim=-1)[1].squeeze(1)
                dec_input = top
                next_id = list(top.clone().cpu().numpy())
                hypo_matrix.append(next_id)
            hypo_matrix = np.array(hypo_matrix).transpose()
            for vec in hypo_matrix:
                sentence = [id2word[id] for id in vec if id2word[id] not in ['<END>', '<PAD>']]
                hypotheses.append(sentence)
        bleu = corpus_bleu(list_of_references=references, hypotheses=hypotheses)
        encoder.train()
        decoder.train()
        return bleu