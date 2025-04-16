# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:01:34 2025

@author: PAUL
"""

def adjust_lr(optimizer, epoch):
    if epoch <= 1100:
        lr = 0.0001
    elif epoch <= 1500:
        lr = 0.0001
    elif epoch <= 2000:
        lr = 0.0001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr