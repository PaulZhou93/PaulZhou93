# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 18:18:00 2025

@author: PAUL
"""

import numpy as np
import torch
import torch.optim as optim
from evaluate_nn import Evaluate
from adjust_lr import adjust_lr
from rmsprop_update import rmsprop_update



def nn_traning(mem,k,mem_size,batch_size,eva,AmpFactor,LinearFactor):

    label = torch.tensor(np.exp(AmpFactor * mem[:,-1:] - LinearFactor))

    params = [p for p in eva.parameters() if p.requires_grad]
    lr = 0
    optimizer = optim.RMSprop(params, lr=lr)
    lr = adjust_lr(optimizer, k)
    total_loss = 0
    for kk in range(mem_size//batch_size):
        norm_s = torch.tensor(mem[batch_size*kk:batch_size*(kk+1),:-1])
        pred = eva(norm_s)
        optimizer.zero_grad()
        loss = torch.mean((pred - label[batch_size*kk:batch_size*(kk+1)]) ** 2)
        total_loss = total_loss + loss.item()
        loss.backward()
        optimizer.step()
    
    ave_loss = total_loss / (mem_size//batch_size)
    
    return eva,ave_loss

def policy_traning(eva,gsw,sq_grads):
    r = eva(gsw)
    # optimizer.zero_grad()
    loss_a = -r
    loss_a.backward(retain_graph=True)
    gsw.grad
    lr_a = rmsprop_update(gsw.grad, sq_grads)
    if np.random.rand() < 0.9:
        gs_n = torch.tensor(gsw + lr_a * gsw.grad, requires_grad = True)
    else:
        gs_n = torch.tensor(gsw + lr_a * torch.randn(8), requires_grad = True)
        
    return gs_n,sq_grads