# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:11:37 2025

@author: PAUL
"""

import numpy as np
import torch
from reward_cal import reward_cal,reward_cal_samp,reward_cal_bit


def random_mem_update(k,sigma2,mem,order,r_mi_max,r_gmi_max,gsw,mi_mode):
    err_rate = 0
    s = np.array([1,3,5,7]) * np.abs(1 + 0.05 * np.random.randn(order))
    exploration_factor = sigma2 ** 0.85 * 0.4 * np.abs(1 + 0.05 * np.random.randn())
    exploration_factor = np.min((exploration_factor,1))
    sp = - s ** 2 * exploration_factor
    # sp = np.exp(sp) / np.sum(np.exp(sp))
    if mi_mode == 0:
        r_mi = reward_cal(s,sp,sigma2)
        if np.random.rand() < err_rate:
            bit_loc = np.random.randint(15)
            if bit_loc == 0:
                r_mi = 0
            else:
                denom = 2 ** (2 - bit_loc)
                bit_value = (r_mi // denom) % 2
                if bit_value == 0:
                    r_mi = r_mi + denom
                else:
                    r_mi = r_mi - denom
            prob = np.exp(gsw.detach().numpy()[4:]) / np.sum(np.exp(gsw.detach().numpy()[4:]))
            max_ent = - np.sum(prob * np.log2(prob))
            r_mi = np.min((r_mi,max_ent))
    else:
        r_mi = reward_cal_samp(s,sp,sigma2)
    r_gmi = reward_cal_bit(s,sp,sigma2)
    mem[k,:order] = s
    mem[k,order:-1] = sp
    mem[k,-1] = r_mi
    if r_mi > r_mi_max:
        r_mi_max = r_mi
        r_gmi_max = r_gmi
        gsw = torch.tensor(np.concatenate((s,sp)).astype(np.float32),requires_grad = True)
    return gsw, r_mi_max, r_gmi_max, mem

def greedy_mem_update(gsw,gs_n,sigma2,r_mi_max,r_gmi_max,k,mem_size,mem,order,mi_mode,eva,ave_loss):
    err_rate = 0.00
    s = gs_n[:order].detach().numpy()
    sp = gs_n[order:].detach().numpy()
    if mi_mode == 0:
        r_mi = reward_cal(s,sp,sigma2)
        if np.random.rand() < err_rate:
            bit_loc = np.random.randint(15)
            if bit_loc == 0:
                r_mi = 0
            else:
                denom = 2 ** (2 - bit_loc)
                bit_value = (r_mi // denom) % 2
                if bit_value == 0:
                    r_mi = r_mi + denom
                else:
                    r_mi = r_mi - denom
            prob = np.exp(gsw.detach().numpy()[4:]) / np.sum(np.exp(gsw.detach().numpy()[4:]))
            max_ent = - np.sum(prob * np.log2(prob))
            r_mi = np.min((r_mi,max_ent))
    else:
        r_mi = reward_cal_samp(s,sp,sigma2)
    r_gmi = reward_cal_bit(s,sp,sigma2)
    eva_now = eva(gsw)
    eva_next = eva(gs_n)
    if r_mi > r_mi_max:
        r_mi_max = r_mi
        r_gmi_max = r_gmi
        gsw = gs_n
        
    addr = (k + 1) % mem_size
    mem[addr,:order] = s
    mem[addr,order:-1] = sp
    mem[addr,-1] = r_mi
    
    return gsw, r_mi_max, r_gmi_max, mem
