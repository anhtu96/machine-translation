# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 21:22:45 2019

@author: tungo
"""

import torch
from torch.utils.data import Dataset

class MTDataset(Dataset):
    """
    Create a Dataset class to feed into a torch DataLoader
    
    Inputs:
    - input_matrix: word vectors of input sentences
    - target_matrix: word vectors of target sentences
    
    Return:
    - pairs of input tensors - target tensors
    """
    def __init__(self, input_matrix, target_matrix):
        self.data = []
        for i in range(len(input_matrix)):
            self.data.append((input_matrix[i], target_matrix[i]))
            
    def __getitem__(self, idx):
        return (torch.Tensor(self.data[idx][0]), torch.Tensor(self.data[idx][1]))
    
    def __len__(self):
        return len(self.data)