#!/usr/bin/env python

import torch
import numpy as np
import sys


def dot(x1, x2):
    return torch.sum(x1*x2)

def ip(x):
    return dot(x, x)

def dot_batch(x1, x2):
    return torch.sum(x1*x2, dim=list(range(1, len(x1.shape))))

def ip_batch(x):
    return dot_batch(x, x)
