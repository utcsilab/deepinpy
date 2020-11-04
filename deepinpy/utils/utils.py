#!/usr/bin/env python

import torch
import numpy as np
import h5py
import scipy

import deepinpy.utils.complex as cp

'''
Utility functions """
'''
# FIXME: Some methods use sub-methods that have optional param axes, axes should also be added to these methods

def h5_write(filename, data):
    """Read numpy arrays from h5 file.

    Args:
        filename (str): Path to the file
        data (dict): Dictionary of data to save
        
    """
    with h5py.File(filename, 'w') as F:
        for key in data.keys():
            F.create_dataset(key, data=data[key])

def h5_read(filename, key_list):
    """Read numpy arrays from h5 file.

    Args:
        filename (str): Path to the file
        key_list (list): List of keys to open

    Returns:
        A dictionary of numpy arrays matching key_list
    """
    data = {}
    with h5py.File(filename, 'r') as F:
        for key in key_list:
            try:
                data[key] = np.array(F[key])
            except:
                print('Key {} not found in {}. Skipping.'.format(key, filename))
                data[key] = None
    return data

# TODO: Unused, potentially depreciated
# FIXME: Dim should be an optional parameter for consistency with torch.topk
def topk(inp, k, dim):
    """Finds the top k values of a vector along a specified dimension.

    Args:
        inp (torch.Tensor): The tensor to retrieve the top k values from.
        k (int): The number of values to retrieve.
        dim (int): The dimension of the Tensor the values should be retrieved from.

    Returns:
        A Tensor of the same shape as the original containing only the top k values.
    """
    _topk, _idx = torch.topk(abs(inp), k, dim=dim)
    _topk = torch.gather(inp, dim, _idx).sign() * _topk
    out = 0*inp
    return out.scatter_(dim, _idx, _topk)

def t2n(x):
    """Converts a 2-channel real Tensor into a numpy array containing complex values.

    Args:
        x (Tensor): The Tensor to be converted.

    Returns:
        A numpy array containing complex-valued information from x.
    """

    return cp.r2c(t2n2(x))

def t2n2(x):
    """Converts a Tensor into a numpy array directly.

    Args:
        x (Tensor): The Tensor to convert.

    Returns:
        A numpy array initialized from x which has been detached from its computational graph.
    """

    return np.array(x.detach().cpu())

def itemize(x):
    """Converts a Tensor into a list of Python numbers.

    Args:
        x (Tensor): The tensor being itemized.

    Returns:
        Python list containing the itemized contents of the tensor.
    """

    if len(x.shape) < 1:
        x = x[None]
    if x.shape[0] > 1:
        return [xx.item() for xx in x]
    else:
        return x.item()

def fftmod(out):
    """Performs a modulated FFT on the input, that is multiplying every other line by exp(j*pi) which is a shift of N/2, hence modulating the output by +/- pi.

    Args:
        out (array_like): Input to the FFT.

    Returns:
        The modulated FFT of the input.
    """

    out2 = out.copy()
    out2[...,::2,:] *= -1
    out2[...,:,::2] *= -1
    out2 *= -1
    return out2

def fftshift(x, axes=(-2, -1)):
    """Shifts the zero-frequency component of the last two dimensions of the input to the center of the spectrum.

    Args:
        x (array_like): The vector to be shifted.
        axes (array_like): The axes along which to apply the shift, default (-2, -1), None uses all axes.

    Returns:
        The shifted version of x.
    """

    return scipy.fftpack.fftshift(x, axes=axes)

def ifftshift(x, axes=(-2, -1)):
    """Removes the effects of shifting the zero-frequency component of the last two dimensions of the input to the center of the spectrum.

    Args:
        x (array_like): The vector whose shift is to be removed.
        axes (array_like): The axes along which to apply the shift, default (-2, -1), None uses all axes.

    Returns:
        An unshifted version of x.
    """

    return scipy.fftpack.ifftshift(x, axes=axes)

def fft2c(x):
    """Performs a 2-dimensional centered FFT on the last two dimensions of the input.

    Args:
        x (array_like): The values to be transformed.

    Returns:
        The 2-dimensional centered FFT of x.
    """

    return fftshift(fft2(ifftshift(x)))

def ifft2c(x):
    """Performs an inverse 2-dimensional centered FFT on the last two dimensions of the input.

    Args:
        x (array_like): The values to be transformed.

    Returns:
        The inverse 2-dimensional centered FFT of x.
    """
    return ifftshift(ifft2(fftshift(x)))

def fft2uc(x):
    """Performs a unitary-centered 2-dimensional FFT on the last two dimensions of the input.

    Args:
        x (array_like): The values to be transformed.

    Returns:
        The unitary-centered 2-dimensional FFT of x.
    """
    return fft2c(x) / np.sqrt(np.prod(x.shape[-2:]))

def ifft2uc(x):
    """Performs an inverse unitary-centered 2-dimensional FFT on the last two dimensions of the input.

    Args:
        x (array_like): The values to be transformed.

    Returns:
        The inverse unitary-centered 2-dimensional FFT of x.
    """

    return ifft2c(x) * np.sqrt(np.prod(x.shape[-2:]))

def fft2(x, axes=(-2, -1)):
    """Performs a 2-dimensional FFT on the last two dimensions of the input.

    Args:
        x (array_like): The values to be transformed.
        axes (array_like): The axes along which to apply the shift, default (-2, -1), None uses all axes.

    Returns:
        The 2-dimensional FFT of x.
    """

    return scipy.fftpack.fft2(x, axes=axes)

def ifft2(x, axes=(-2, -1)):
    """Performs an inverse 2-dimensional FFT on the last two dimensions of the input.

    Args:
        x (array_like): The values to be transformed.
        axes (array_like): The axes along which to apply the shift, default (-2, -1), None uses all axes.

    Returns:
        The inverse 2-dimensional FFT of x.
    """

    return scipy.fftpack.ifft2(x, axes=axes)
