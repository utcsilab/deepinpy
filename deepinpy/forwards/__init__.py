"""
Deep inverse problems in Python

forwards submodule
A Forward object takes an input image x and returns measurements y
"""

from .mcmri.mcmri import MultiChannelMRI, fft_forw, fft_adj
from .mcmri.dataset import MultiChannelMRIDataset
