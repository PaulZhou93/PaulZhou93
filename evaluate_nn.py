# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:49:36 2025

@author: PAUL
"""

import torch.nn as nn

class Evaluate(nn.Module):

    def __init__(self):
        super().__init__()
        width = 200
        self.mlp = nn.Sequential(
            nn.Linear(8, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        output = self.mlp(x)
    
        return output