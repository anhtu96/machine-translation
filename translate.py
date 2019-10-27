# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:25:52 2019

@author: tungo
"""
import string
import torch

def translate(sentence, inp_word2id, trg_word2id, trg_id2word, encoder, decoder, trg_max_len, device='cpu'):
    """
    Generate translation for input sentence.
    
    Inputs:
    - sentence: a sentence in string format
    - inp_word2id: word2id from input training set
    - trg_word2id: word2id from target training set
    - trg_id2word: id2word from target training set
    - encoder, decoder: Encoder, Decoder models
    - trg_max_len: max length for target sentence
    - device: 'cpu' or 'cuda'
    
    Return a sentence
    """
    exclude = list(string.punctuation) + list(string.digits)
    sentence = '<START> ' + ''.join([char for char in sentence if char not in exclude]).strip().lower() + ' <END>'
    sen_matrix = [inp_word2id[s] for s in sentence.split()]
    sen_tensor = torch.Tensor(sen_matrix).to(device=device, dtype=torch.long).unsqueeze(0)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        enc_out, enc_hidden = encoder(sen_tensor)
        dec_hidden = enc_hidden
        dec_input = torch.Tensor([trg_word2id['<START>']]).to(device='cuda', dtype=torch.long)
        output_list = []
        for t in range(1, trg_max_len):
            out, dec_hidden = decoder(dec_input, dec_hidden, enc_out)
            dec_input = torch.max(out, dim=-1)[1].squeeze(1)
            next_id = dec_input.squeeze().clone().cpu().numpy()
            next_word = trg_id2word[next_id]
            if next_word == '<END>':
                break
            output_list.append(next_word)
        return ' '.join(output_list)