#!/usr/bin/env python
"""Vector operations for use in calculating conjugate gradient descent."""

import torch
import numpy as np


def dot(x1, x2):
    """Finds the dot product of two vectors.

    Args:
        x1 (Tensor): The first input vector.
        x2 (Tensor): The second input vector.

    Returns:
        The dot product of x1 and x2.
    """

    return torch.sum(x1*x2)

def ip(x):
    """Finds the identity product of a vector.

    Args:
        x (Tensor): The input vector.

    Returns:
        The identity product of x.
    """

    return dot(x, x)

def dot_batch(x1, x2):
    """Finds the dot product of two multidimensional Tensors holding batches of data.

    Args:
        x1 (Tensor): The first multidimensional Tensor.
        x2 (Tensor): The second multidimensional Tensor.

    Returns:
        The dot products along each dimension of x1 and x2.
    """

    batch = x1.shape[0]
    return torch.reshape(x1*x2, (batch, -1)).sum(1)

def ip_batch(x):
    """Finds the identity product of a multidimensional Tensor holding a batch of data.

    Args:
        x (Tensor): The tensor who’s batch identity product will be computed.

    Returns:
        The batch identity product of x.
    """

    return dot_batch(x, x)

def l2ball_proj_batch(x, eps):
    """ Performs a batch projection onto the L2 ball.

    Args:
        x (Tensor): The tensor to be projected.
        eps (Tensor): A tensor containing epsilon values for each dimension of the L2 ball.

    Returns:
        The projection of x onto the L2 ball.
    """

    #print('l2ball_proj_batch')
    reshape = (-1,) + (1,) * (len(x.shape) - 1)
    q1 = ip_batch(x).sqrt()
    #print(eps,q1)
    q1_clamp = torch.min(q1, eps)

    z = x * q1_clamp.reshape(reshape) / (1e-8 + q1.reshape(reshape))
    #q2 = ip_batch(z).sqrt()
    #print(eps,q1,q2)
    return z
