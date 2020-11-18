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

def dot_single(x):
    """Finds the dot product of a vector with itself

    Args:
        x (Tensor): The input vector.

    Returns:
        The dot product of x and x.
    """

    return dot(x, x)

def dot_batch(x1, x2):
    """Finds the dot product of two multidimensional Tensors, preserving the batch dimension.

    Args:
        x1 (Tensor): The first multidimensional Tensor.
        x2 (Tensor): The second multidimensional Tensor.

    Returns:
        The dot products along each dimension of x1 and x2.
    """

    batch = x1.shape[0]
    return torch.reshape(x1*x2, (batch, -1)).sum(1)


def dot_single_batch(x):
    """Finds the dot product of a multidimensional Tensors with itself, preserving the batch dimension.

    Args:
        x (Tensor): The multidimensional Tensor.

    Returns:
        The dot products along each non-batch dimension of x and x.
    """

    return dot_batch(x, x)

def zdot(x1, x2):
    """Finds the complex-valued dot product of two complex-valued vectors.

    Args:
        x1 (Tensor): The first input vector.
        x2 (Tensor): The second input vector.

    Returns:
        The dot product of x1 and x2, defined as sum(conj(x1) * x2)
    """

    return torch.sum(torch.conj(x1)*x2)

def zdot_single(x):
    """Finds the complex-valued dot product of a complex-valued vector with itself

    Args:
        x (Tensor): The input vector.

    Returns:
        The dot product of x and x., defined as sum(conj(x) * x)
    """

    return zdot(x, x)

def zdot_batch(x1, x2):
    """Finds the complex-valued dot product of two complex-valued multidimensional Tensors, preserving the batch dimension.

    Args:
        x1 (Tensor): The first multidimensional Tensor.
        x2 (Tensor): The second multidimensional Tensor.

    Returns:
        The dot products along each dimension of x1 and x2.
    """

    batch = x1.shape[0]
    return torch.reshape(torch.conj(x1)*x2, (batch, -1)).sum(1)


def zdot_single_batch(x):
    """Finds the complex-valued dot product of a multidimensional Tensors with itself, preserving the batch dimension.

    Args:
        x (Tensor): The multidimensional Tensor.

    Returns:
        The dot products along each non-batch dimension of x and x.
    """

    return zdot_batch(x, x)

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
    x = x.contiguous()
    q1 = torch.real(zdot_single_batch(x)).sqrt()
    #print(eps,q1)
    q1_clamp = torch.min(q1, eps)

    z = x * q1_clamp.reshape(reshape) / (1e-8 + q1.reshape(reshape))
    #q2 = torch.real(zdot_single_batch(z)).sqrt()
    #print(eps,q1,q2)
    return z
