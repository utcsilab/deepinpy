#!/usr/bin/env python

import torch
import numpy as np


def dot(x1, x2):
    return torch.sum(x1*x2)

def ip(x):
    return dot(x, x)

def dot_batch(x1, x2):
    return torch.sum(x1*x2, dim=list(range(1, len(x1.shape))))

def ip_batch(x):
    return dot_batch(x, x)

def l2ball_proj_batch(x, eps):
    #print('l2ball_proj_batch')
    reshape = (-1,) + (1,) * (len(x.shape) - 1)
    q1 = ip_batch(x).sqrt()
    #print(eps,q1)
    q1_clamp = torch.min(q1, eps)

    z = x * q1_clamp.reshape(reshape) / (1e-8 + q1.reshape(reshape))
    #q2 = ip_batch(z).sqrt()
    #print(eps,q1,q2)
    return z
