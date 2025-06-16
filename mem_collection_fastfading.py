# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:28:04 2025

@author: PAUL
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:11:37 2025

@author: PAUL
"""

import numpy as np
import torch
from reward_cal import reward_cal,reward_cal_samp,reward_cal_bit


def random_mem_update(k,sigma2,mem,order,r_mi_max,r_gmi_max,gsw,mi_mode):
    s = np.array([1,3,5,7]) * np.abs(1 + 0.05 * np.random.randn(order))
    exploration_factor = sigma2 ** 0.85 * 0.4 * np.abs(1 + 0.05 * np.random.randn())
    sp = - s ** 2 * exploration_factor
    # sp = np.exp(sp) / np.sum(np.exp(sp))
    
    amp_gain = np.arange(0.1,3.1,0.1)
    p_g = 2 * amp_gain * np.exp(- amp_gain ** 2)
    r_mi = 0
    r_gmi = 0
    for k in range(len(amp_gain)):
        inst_sigma2 = sigma2 / (amp_gain[k] ** 2)
        if mi_mode == 0:
            r_mi = r_mi + p_g[k] * reward_cal(s,sp,inst_sigma2)
        else:
            r_mi = r_mi + p_g[k] * reward_cal_samp(s,sp,inst_sigma2)
        r_gmi = r_gmi + p_g[k] * reward_cal_bit(s,sp,inst_sigma2)
        
    r_mi = r_mi / 10
    r_gmi = r_gmi / 10    
    mem[k,:order] = s
    mem[k,order:-1] = sp
    mem[k,-1] = r_mi
    if r_mi > r_mi_max:
        r_mi_max = r_mi
        r_gmi_max = r_gmi
        gsw = torch.tensor(np.concatenate((s,sp)).astype(np.float32),requires_grad = True)
    return gsw, r_mi_max, r_gmi_max, mem

def greedy_mem_update(gsw,gs_n,sigma2,r_mi_max,r_gmi_max,k,mem_size,mem,order,mi_mode):
    s = gs_n[:order].detach().numpy()
    sp = gs_n[order:].detach().numpy()
    
    amp_gain = np.arange(0.1,3.1,0.1)
    p_g = 2 * amp_gain * np.exp(- amp_gain ** 2)
    r_mi = 0
    r_gmi = 0
    for k in range(len(amp_gain)):
        inst_sigma2 = sigma2 / (amp_gain[k] ** 2)
        if mi_mode == 0:
            r_mi = r_mi + p_g[k] * reward_cal(s,sp,inst_sigma2)
        else:
            r_mi = r_mi + p_g[k] * reward_cal_samp(s,sp,inst_sigma2)
        r_gmi = r_gmi + p_g[k] * reward_cal_bit(s,sp,inst_sigma2)
    r_mi = r_mi / 10
    r_gmi = r_gmi / 10
    if r_mi > r_mi_max:
        r_mi_max = r_mi
        r_gmi_max = r_gmi
        gsw = gs_n
        
    addr = (k + 1) % mem_size
    mem[addr,:order] = s
    mem[addr,order:-1] = sp
    mem[addr,-1] = r_mi
    
    return gsw, r_mi_max, r_gmi_max, mem
