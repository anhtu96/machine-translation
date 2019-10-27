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
    Create a language class that contains necessary attributes.
    
    Inputs:
    - sentence_list: a list containing all sentences (string format)
    - train: True if used for training phase, otherwise False
    - word2id, id2word: get word2id and id2word from existing training set. Only used for val/test set (train = False), ignored if train = True.
    
    Returns a class that contains:
    - max_len: length of longest sentence in the list
    - sentences: list containing all sentences
    - word2id, id2word
    - vocab_size: number of words after preprocessing
    - wordvec: word vectors
    """
    def __init__(self, sentence_list, train=True, word2id=None, id2word=None):
        self.word2id = word2id
        self.id2word = id2word
        self.train = train
        self.preprocess(sentence_list)
        self.get_vocab()
        self.get_word_vectors()
        
    def preprocess(self, sentence_list):
        """
        Preprocess sentences by adding <START> and <END> tokens, then padding all sentences to the same length with <PAD> tokens.
        """
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
        """
        Retrieve word2id, id2word, vocab size.
        """
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
        """
        Retrieve word vectors.
        """
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