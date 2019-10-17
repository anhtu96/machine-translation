# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 23:03:27 2019

@author: tungo
"""
import string
import numpy as np
import random

exclude = list(string.punctuation) + list(string.digits)

class Language(object):
    """
    Create a class that contains necessary attributes for training
    
    Input: name of text file containing sentences
    
    Returns a class that contains:
    - max_len
    - sentences
    - word2id, id2word
    - vocab_size
    - wordvec
    """
    def __init__(self, sentence_list, train=True, word2id=None, id2word=None):
        self.word2id = word2id
        self.id2word = id2word
        self.train = train
        self.preprocess(sentence_list)
        self.get_vocab()
        self.get_word_vectors()
        
    def preprocess(self, sentence_list):
        self.max_len = 0
        self.sentences = []
        for sen in sentence_list:
            sen = '<START> ' + sen + ' <END>'
            length = len(sen.split())
            self.sentences.append(sen)
            if self.max_len < length:
                self.max_len = length
        self.padding()
    
    def padding(self):
        """
        Extend all sentences to the same size by adding <PAD> tokens.
        """
        for i, sen in enumerate(self.sentences):
            length = len(sen.split())
            diff = self.max_len - length
            paddings = [' <PAD>'] * diff
            self.sentences[i] = sen + ''.join(paddings)
            
    def get_vocab(self):
        if self.train:
            self.word2id = {}
            self.id2word = []
            for s in self.sentences:
                for char in s.split():
                    if char not in self.word2id:
                        self.id2word.append(char)
                        self.word2id[char] = len(self.id2word) - 1
        self.vocab_size = len(self.id2word)
        
    def get_word_vectors(self):
        self.wordvec = []
        for i, sen in enumerate(self.sentences):
            id_list = []
            for s in sen.split():
                if s in self.word2id:
                    id_list.append(self.word2id[s])
                else:
                    id_list.append(random.randint(0, self.vocab_size-1))
            self.wordvec.append(id_list)
        self.wordvec = np.array(self.wordvec)