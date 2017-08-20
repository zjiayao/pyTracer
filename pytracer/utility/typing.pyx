"""
typing.py

pytracer.utility package

Define types with Cython.

Created by Jiayao on Aug 20, 2017
"""
from __future__ import (division, absolute_import)
import numpy as np
cimport numpy as np

FLOAT = np.float32
ctypedef np.float32_t FLOAT_t

INT = np.int32
ctypedef np.int32_t INT_t

UINT = np.uint32
ctypedef np.uint32_t UINT_t



