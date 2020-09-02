# DeepInPy -- Getting Started
This document is intended to show how to get started with a training experiment. 

In the current setup, the training is driven by a __data__ file and a __config__ file. The data file specifies the training data. the config file specifies the training procedure. Right now, the config file has options for every training procedure implemented, even though many are only relevant to a specific training algorithm, model, etc.

## Data format:
DeepInPy excepts a complex-valued, multi-channel MRI format. Even when the data are single-coil, the format should be followed.

The data format is a h5 file, consisting of the following fields:
```bash
imgs: [Ntraining, N1, N2, ..., NT, X, Y, Z]: np.complex
masks: [Ntraining, N1, N2, ..., NT, X, Y, Z]: np.float
maps: [Ntraining, Ncoil, N1, N2, ..., NT, X, Y, Z]: np.complex
ksp: [Ntraining, Ncoil, N1, N2, ..., NT, X, Y, Z]: np.complex
```

`Ntraining` is the number of training examples. If `Ntraining=1`, it should still be included as a singleton dimension.  
`N1`, `N2, ..., `NT` are higher-order dimensions, and can be used for multi-phase data (e.g. temporal, contrast, coefficients, phases, etc.). These dimensions are optional and can be excluded
`X`, `Y`, `Z` are spatial dimensions. In the case of 2D data, the `Z` dimension can be excluded.

Except for the masks, all data should be stored as complex-valued arrays.

### Example: 2D 8-coil data with 100 training examples
```bash
imgs: [100, 256, 256]: np.complex
masks: [100, 256, 256]: np.float
maps: [100, 8, 256, 256]: np.complex
ksp: [100, 8, 256, 256]: np.complex
```

### Example: 2D single-coil data with 1 training example
```bash
imgs: [1, 256, 256]: np.complex
masks: [1, 256, 256]: np.float
maps: [1, 1, 256, 256]: np.complex
ksp: [1, 1, 256, 256]: np.complex
```
Note that the `maps` array can be all-ones in this case

### Example: 2D 8-coil data with 100 training examples and 20 temporal phases
```bash
imgs: [100, 20, 256, 256]: np.complex
masks: [100, 20, 256, 256]: np.float
maps: [100, 8, 20, 256, 256]: np.complex
ksp: [100, 8, 20, 256, 256]: np.complex
```
Note that the `maps` array can be all-ones in this case

### Example: 2D 8-coil data, solving for each channel separately
We use the same interface by treating the coil dimension as a higher-order dimension, and creating an all-ones maps array.
We tell the code that it is "one-channel" data with a higher-order dimension equal to 8
```bash
imgs: [100, 8, 256, 256]: np.complex
masks: [100, 8, 256, 256]: np.float
maps: [100, 1, 8, 256, 256]: np.complex
ksp: [100, 1, 8, 256, 256]: np.complex
```

### Writing/reading data file
To write a data file, you can use the `deepinpy.utils.utils.h5_write` function. The function takes the path to the target h5 file and a dictionary of key-value pairs:
```python
# example data writer for 2D images with 10 training examples and 8 coils

from deepinpy.utils.utils import h5_write

imgs = np.random.randn(10, 256, 256, dtype=np.complex)
masks = np.random.randn(10, 256, 256, dtype=np.complex)
maps = np.random.randn(10, 8, 256, 256, dtype=np.float)
ksp = np.random.randn(10, 8, 256, 256, dtype=np.complex)

data = {'imgs': imgs, 'masks': masks, 'maps': maps, 'ksp': ksp}

h5_write('mydata.h5', data)
```

There is also a similar `h5_read` function to load the training set.
