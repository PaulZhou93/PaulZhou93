# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:03:57 2025

@author: PAUL
"""

import numpy as np
import torch

def rmsprop_update(grad, sq_grads, lr=0.0001, beta=0.99, epsilon=1e-8):
    lr = np.array([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001])
    lr_new = torch.zeros(len(grad))
    for k in range(len(grad)):
        sq_grads[k] = beta * sq_grads[k] + (1 - beta) * (grad[k] ** 2)
        lr_new[k] = lr[k] / (np.sqrt(sq_grads[k]) + epsilon)
    return lr_new