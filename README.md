# PyRTSTools 
![version](https://img.shields.io/github/manifest-json/v/linto-ai/sfeatpy/release)   [![pypi version](https://img.shields.io/pypi/v/sfeatpy)](https://pypi.org/project/sfeatpy/)
## Introduction

Python library to extract MFCC parameters.

## Installation

### pypi

```bash
pip install sfeatpy
```

### From source

```bash
git clone https://github.com/linto-ai/sfeatpy.git
cd sfeatpy
./setup.py install
```

## Usage

```python
import sfeatpy
import numpy as np

rd_signal = np.random.random(16000)

res = sfeatpy.mfcc(rd_signal,           # audio signal
                   sample_rate,         # sample_rate -- Audio sampling rate (default 16000)  
                   window_length,       # window_length -- window size in sample (default 1024)  
                   window_stride,       # window_stride -- window stride in sample (default 512)  
                   fft_size,            # fft_size -- fft number of points (default 1024) 
                   min_freq,            # min_freq -- minimum frequency in hertz (default 20) 
                   max_freq,            # max_freq -- maximum frequency in hertz (default 7000) 
                   num_filter,          # num_filter -- number of MEL bins (default 40) 
                   num_coef,            # num_coef -- number of output coeficients (default 20) 
                   windowFun,           # windowFun -- window function: 0- None | 1- hamming (default 0) 
                   preEmp,              # preEmp -- preEmphasis factor ignored on None (default 0.97) 
                   keep_first_value     # keep_first_value -- if False discard first MFCC value (default False)
                   )
res.shape
> (30,20)

```

## Limitations

* Values are not checked to keep the processing efficient.
* Works only on Mono-channel signal

## Licence
This project is under aGPLv3 licence, feel free to use and modify the code under those terms.
See LICENCE

## Used libraries

* [Numpy](http://www.numpy.org/)
* [Scipy](https://github.com/tensorflow/tensorflow)
