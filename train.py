# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:08:10 2019

@author: tungo
"""
from eval import validate
import torch

def train(encoder, decoder, train_loader, val_loader, optimizer, criterion, id2word, lr_scheduler=None, num_epochs=1, print_every=100, device='cpu', early_stop=False):
    """
    Function for training
    
    Inputs:
    - encoder, decoder
    - train_loader, val_loader: DataLoader for training set and validation set
    - optimizer: a torch.optim optimizer (e.g. torch.optim.Adam(...))
    - criterion: loss function (e.g. nn.CrossEntropyLoss())
    - id2word: id2word for target training set
    - lr_scheduler: learning rate scheduler (e.g. torch.optim.lr_scheduler.StepLR)
    - num_epochs
    - print_every
    - device: 'cpu' or 'cuda'
    """
    encoder.train()
    decoder.train()
    best_bleu = 0
    best_statedict = {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}
    for epoch in range(num_epochs):
        print('Epoch ', epoch + 1)
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device=device, dtype=torch.long)
            y = y.to(device=device, dtype=torch.long)
            enc_out, enc_hidden = encoder(x)
            dec_hidden = enc_hidden
            dec_input = y[:, 0]
            loss = 0
            optimizer.zero_grad()
            for t in range(1, y.size(1)):
                out, dec_hidden = decoder(dec_input, dec_hidden, enc_out)
                dec_input = y[:, t]
                loss += criterion(out.squeeze(1), y[:, t])
            loss.backward()
            optimizer.step()
            if i % print_every == 0:
                print('Iter %d, loss = %f' %(i, loss.item() / y.size(1)))
        if lr_scheduler != None:
            lr_scheduler.step()
        bleu = validate(val_loader, encoder, decoder, id2word, device)
        print('Validation BLEU score: %f\n' %bleu)
        if bleu > best_bleu:
            best_statedict = {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}
            best_bleu = bleu
        elif early_stop:
            print('=== BLEU begins to decrease, training exits ===')
            return best_statedict
    return best_statedict