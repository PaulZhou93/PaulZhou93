# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:52:04 2025

@author: PAUL
"""

import numpy as np


def reward_cal(s,sp,sigma2,resolution = 0.001):
    # sp = np.array([0.2,0.3,0.3,0.2])
    sp = np.exp(sp) / np.sum(np.exp(sp))
    s = s / np.sqrt(np.dot(sp,(s**2)))
    s = np.hstack((-s,s))
    sp = np.hstack((sp/2,sp/2))
    x = np.arange(-10,10,resolution)
    # sp = 1 / order * np.ones(order)
    py = np.zeros(len(x))
    
    for k in range(len(s)):
        py = py + sp[k] / np.sqrt(2*np.pi*sigma2) * (np.exp(- ((x - s[k]) ** 2) / 2 / sigma2))
              
    hy = - resolution * np.sum(py[py!=0] * np.log2(py[py!=0]))
    hxy = 1 / 2*np.log2(2*np.pi*np.exp(1)*sigma2)
    
    r = np.max((hy - hxy, 0))                                                  # Subtracting H(Y|X) is for normalization, but it does not make a difference if not subtracted.
    
    return r

def reward_cal_samp(s,sp,sigma2,resolution = 0.001):
    sp = np.exp(sp) / np.sum(np.exp(sp))
    num = 1000000
    noise = np.sqrt(sigma2) * np.random.randn(num)
    s = s / np.sqrt(np.dot(sp,(s**2)))
    s = np.hstack((-s,s))
    sp = np.hstack((sp/2,sp/2))
    
    s_sa = np.random.choice(s,num,p=sp)
    yy = s_sa + noise
    [a,b] = np.histogram(yy,1000)
    pa = a / num
    interval = np.mean(b[1:] - b[:-1])
    
    hyy = - np.sum(pa[pa!=0] * np.log2(pa[pa!=0]/interval))
    hxy = 1 / 2*np.log2(2*np.pi*np.exp(1)*sigma2)
    
    r = np.max((hyy - hxy, 0))                                                 # Subtracting H(Y|X) is for normalization, but it does not make a difference if not subtracted.
    
    return r

def reward_cal_bit(s,sp,sigma2,resolution = 0.001):
    sp = np.exp(sp) / np.sum(np.exp(sp))
    s = s / np.sqrt(np.dot(sp,(s**2)))
    s = np.hstack((-s,s))
    sp = np.hstack((sp/2,sp/2))
    x = np.arange(-10,10,resolution)
    # sp = 1 / order * np.ones(order)
    py = np.zeros(len(x))
    
    for k in range(len(s)):
        py = py + sp[k] / np.sqrt(2*np.pi*sigma2) * (np.exp(- ((x - s[k]) ** 2) / 2 / sigma2))
              
    hy = - resolution * np.sum(py[py!=0] * np.log2(py[py!=0]))
    
    ind1 = np.array([[0,1,2,3],[4,5,6,7]])
    px1_y_1 = np.zeros(len(x))
    px1_y_2 = np.zeros(len(x))
    p_pre_1 = np.sum(sp[ind1],axis=1)
    for k in range(4):
        px1_y_1 = px1_y_1 + sp[ind1[0,k]] / p_pre_1[0] / np.sqrt(2*np.pi*sigma2) * (np.exp(- ((x - s[ind1[0,k]]) ** 2) / 2 / sigma2))
        px1_y_2 = px1_y_2 + sp[ind1[1,k]] / p_pre_1[1] / np.sqrt(2*np.pi*sigma2) * (np.exp(- ((x - s[ind1[1,k]]) ** 2) / 2 / sigma2))
              
    hx1_y = - p_pre_1[0] * resolution * np.sum(px1_y_1[px1_y_1!=0] * np.log2(px1_y_1[px1_y_1!=0])) - p_pre_1[1] * resolution * np.sum(px1_y_2[px1_y_2!=0] * np.log2(px1_y_2[px1_y_2!=0]))
    
    ind1 = np.array([[0,1,4,5],[2,3,6,7]])
    px1_y_1 = np.zeros(len(x))
    px1_y_2 = np.zeros(len(x))
    p_pre_2 = np.sum(sp[ind1],axis=1)
    for k in range(4):
        px1_y_1 = px1_y_1 + sp[ind1[0,k]] / p_pre_2[0] / np.sqrt(2*np.pi*sigma2) * (np.exp(- ((x - s[ind1[0,k]]) ** 2) / 2 / sigma2))
        px1_y_2 = px1_y_2 + sp[ind1[1,k]] / p_pre_2[1] / np.sqrt(2*np.pi*sigma2) * (np.exp(- ((x - s[ind1[1,k]]) ** 2) / 2 / sigma2))
              
    hx2_y = - p_pre_2[0] * resolution * np.sum(px1_y_1[px1_y_1!=0] * np.log2(px1_y_1[px1_y_1!=0])) - p_pre_2[1] * resolution * np.sum(px1_y_2[px1_y_2!=0] * np.log2(px1_y_2[px1_y_2!=0]))
    
    ind1 = np.array([[0,3,4,7],[1,2,5,6]])
    px1_y_1 = np.zeros(len(x))
    px1_y_2 = np.zeros(len(x))
    p_pre_3 = np.sum(sp[ind1],axis=1)
    for k in range(4):
        px1_y_1 = px1_y_1 + sp[ind1[0,k]] / p_pre_3[0] / np.sqrt(2*np.pi*sigma2) * (np.exp(- ((x - s[ind1[0,k]]) ** 2) / 2 / sigma2))
        px1_y_2 = px1_y_2 + sp[ind1[1,k]] / p_pre_3[1] / np.sqrt(2*np.pi*sigma2) * (np.exp(- ((x - s[ind1[1,k]]) ** 2) / 2 / sigma2))
              
    hx3_y = - p_pre_3[0] * resolution * np.sum(px1_y_1[px1_y_1!=0] * np.log2(px1_y_1[px1_y_1!=0])) - p_pre_3[1] * resolution * np.sum(px1_y_2[px1_y_2!=0] * np.log2(px1_y_2[px1_y_2!=0]))
    
    hxy = 1 / 2*np.log2(2*np.pi*np.exp(1)*sigma2)
    
    pxy = 1 / np.sqrt(2*np.pi*sigma2) * (np.exp(- ((x - s[ind1[0,k]]) ** 2) / 2 / sigma2))
    hxy1 = - resolution * np.sum(pxy[pxy!=0] * np.log2(pxy[pxy!=0]))
    
    h1 = np.sum(- p_pre_1 * np.log2(p_pre_1))
    h2 = np.sum(- p_pre_2 * np.log2(p_pre_2))
    h3 = np.sum(- p_pre_3 * np.log2(p_pre_3))
    
    ht = np.sum(- sp * np.log2(sp))
    
    r = np.max((3*hy - hx1_y -hx2_y - hx3_y + ht - h1 - h2 - h3, 0))
    
    return r