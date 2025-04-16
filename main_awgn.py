# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:22:19 2024

@author: PAUL
"""

import numpy as np
import torch
from evaluate_nn import Evaluate
from mem_collection import random_mem_update,greedy_mem_update
from train import nn_traning,policy_traning

seed = 20241231
order = 4
EsN0 = np.array([11,18])
mi_mode = 0
batch_size = 50
mem_size = 1000
AmpFactor = 20
total_epi = 3000


s_data = np.zeros((len(EsN0),2*order))
r_data = np.zeros(len(EsN0))

for k_e in range(len(EsN0)):

    sigma2 = 10 ** (- EsN0[k_e] / 10)
    mem = np.zeros([mem_size,2*order + 1]).astype(np.float32)
    LinearFactor = AmpFactor / 2 * np.log2(1+1/sigma2)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    mem = mem.astype(np.float32)
    eva = Evaluate()
   
    sq_grads = np.zeros(2*order)
    gsw = torch.tensor([1,3,5,7,1,1,1,1])
    r_mi_max = 0
    r_gmi_max = 0
    
    for k in range(total_epi):
        
        if k < mem_size:
                
            gsw, r_mi_max, r_gmi_max, mem = random_mem_update(k,sigma2,mem,order,r_mi_max,r_gmi_max,gsw,mi_mode)
            if k % 100 == 0:
                print("episode: ",k,"reward: ", 2 * r_gmi_max)
            
        else:
            
            eva,ave_loss = nn_traning(mem,k,mem_size,batch_size,eva,AmpFactor,LinearFactor)
            gs_n,sq_grads = policy_traning(eva,gsw,sq_grads)
            gsw, r_mi_max, r_gmi_max, mem = greedy_mem_update(gsw,gs_n,sigma2,r_mi_max,r_gmi_max,k,mem_size,mem,order,mi_mode,eva,ave_loss)
        
            if k % 100 == 0:
                print("episode: ",k,"loss: ",ave_loss,"reward: ", 2 * r_gmi_max)
    s_data[k_e] = gsw.detach().numpy()
    r_data[k_e] = r_gmi_max
        
