"""
__init__.py

pytracer package

Created by Jiayao on Aug 13, 2017
"""
from __future__ import (absolute_import, print_function, division)
import numpy as np
import scipy.spatial  # cKdTree

# Type Alias
KdTree = scipy.spatial.cKDTree

# Global Constant
INT = int
UINT = np.uint32
FLOAT = np.float64
DOUBLE = np.float64
FILTER_TABLE_SIZE = 16

# Global Constants
EPS = 1e-5
HANDEDNESS = 'left'
PI = FLOAT(np.pi)
INV_PI = FLOAT(1. / np.pi)
INV_2PI = FLOAT(1. / (2. * np.pi))

# Global Static
IrIsotropicData = {}
ReHalfangleData = {}

# Utility functions
import pytracer.utility as util

# Spectrum
from pytracer.spectral import (Spectrum, SpectrumType)
