# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:25:52 2019

@author: tungo
"""
import string
import torch

def translate(sentence, inp_word2id, trg_word2id, trg_id2word, encoder, decoder, inp_max_len, trg_max_len, device='cpu'):
    exclude = list(string.punctuation) + list(string.digits)
    sentence = '<START> ' + ''.join([char for char in sentence if char not in exclude]).strip().lower() + ' <END>'
    length = len(sentence.split())
    diff = inp_max_len - length
    sentence = sentence + ''.join([' <PAD>']*diff)
    sen_matrix = [inp_word2id[s] for s in sentence.split()]
    sen_tensor = torch.Tensor(sen_matrix).to(device=device, dtype=torch.long).unsqueeze(0)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        enc_out, enc_hidden_backward, enc_hidden_forward = encoder(sen_tensor)
        dec_hidden = enc_hidden_backward
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