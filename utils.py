# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:36:36 2019

@author: tungo
"""
import string
import random
import numpy as np
import torch

def generate_seed(seed):
    """
    Generate a seed for deterministic random calculation.
    
    Input:
    - seed: an integer for seed
    """
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess(inp_filename, target_filename, max_len):
    with open(inp_filename, 'r', encoding='utf8') as f_inp:
        lines_inp = f_inp.read().split('\n')
    with open(target_filename, 'r', encoding='utf8') as f_trg:
        lines_trg = f_trg.read().split('\n')
    
    sentences_inp, sentences_trg = [], []
    exclude = list(string.punctuation) + list(string.digits)
    
    for sen_inp, sen_trg in zip(lines_inp, lines_trg):
        sen_inp = ''.join([char for char in sen_inp if char not in exclude]).strip().lower()
        sen_trg = ''.join([char for char in sen_trg if char not in exclude]).strip().lower()
        len_inp = len(sen_inp.split())
        len_trg = len(sen_trg.split())
        if len_inp <= max_len and len_trg <= max_len:
            sentences_inp.append(sen_inp)
            sentences_trg.append(sen_trg)
    f_inp.close()
    f_trg.close()
    return sentences_inp, sentences_trg