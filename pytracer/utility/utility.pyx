"""
utility.py

pytracer.utility package

Cython implementation of utility
functions.

Created by Jiayao on Aug 13, 2017
"""
from __future__ import (division, absolute_import)
import numpy as np
cimport numpy as np
import scipy.spatial

from pytracer.utility.typing import (FLOAT, INT, UINT)

__all__ = ['progress_reporter','logging','feq', 'eq_unity', 'ne_unity',
           'ftoi', 'ctoi', 'rtoi', 'lerp', 'round_pow_2', 'next_pow_2', 'is_pow_2', 'ufunc_lerp', 'clip']

# Type Alias
KdTree = scipy.spatial.cKDTree

# Global Constants
EPS = 1e-5
HANDEDNESS = 'left'
FILTER_TABLE_SIZE = 16
PI = FLOAT(np.pi)
INV_PI = FLOAT(1. / np.pi)
INV_2PI = FLOAT(1. / (2. * np.pi))

# Global Static
IrIsotropicData = {}
ReHalfangleData = {}


# Global Functions
def logging(tp: str, msg: str):
	print("[{}] {}".format(tp.upper(), msg))


def progress_reporter(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
	"""
	Call in a loop to create terminal progress bar
	:params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
	"""
	# print()
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filled_length = int(length * iteration // total)
	bar = fill * filled_length + '-' * (length - filled_length)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
	# Print New Line on Complete
	if iteration == total:
		percent = ("{0:." + str(decimals) + "f}").format(100)
		bar = fill * length
		print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))


cdef inline FLOAT fmin(FLOAT a, FLOAT b):
	return a if a <= b else b

cdef inline FLOAT fmax(FLOAT a, FLOAT b):
	return a if a >= b else b

cdef inline bool feq(FLOAT a, FLOAT b):
	return a - b > -EPS and a - b < EPS

cdef inline bool eq_unity(x: FLOAT):
	return feq(x, )

def eq_unity(x: (float, FLOAT)):
	"""Equality test with unity."""
	return (x > 1. - EPS) and (x < 1. + EPS)


def ne_unity(x: (float, FLOAT)) -> bool:
	"""Inequality test with unity."""
	return x < 1. - EPS or x > 1. + EPS


def is_zero(x: (float, FLOAT)) -> bool:
	"""Equality test with zero."""
	return x > -EPS and x < EPS


def not_zero(x: (float, FLOAT)) -> bool:
	"""Inequality test with zero"""
	return x < -EPS or x > EPS


def ftoi(x: (float, FLOAT)) -> INT:
	"""Floor to integer"""
	return INT(np.floor(x))


def ctoi(x: (float, FLOAT)) -> INT:
	"""Ceiling to integer"""
	return INT(np.ceil(x))


def rtoi(x: (float, FLOAT)) -> INT:
	"""Round to integer"""
	return INT(np.round(x))


def lerp(t: (float, FLOAT), v1: (float, FLOAT), v2: (float, FLOAT)) -> (float, FLOAT):
	"""Linear interpolation between `v1` and `v2`"""
	return (1. - t) * v1 + t * v2


def round_pow_2(x: INT) -> INT:
	"""Round to nearest power of 2"""
	return INT(2 ** np.round(np.log2(x)))


def next_pow_2(x: INT) -> INT:
	"""Round to next(or current) power of 2"""
	return INT(2 ** np.ceil(np.log2(x)))


def is_pow_2(x: INT) -> bool:
	"""Test whether is power of 2"""
	return x & (x-1) == 0
	# return True if x == 0 else (np.log2(x) % 1) == 0.


def clip(x, min=0., max=np.inf):
	from pytracer import Spectrum

	if isinstance(x, (float, FLOAT, int, INT, UINT)):
		return np.clip(x, min, max).astype(type(x))
	elif isinstance(x, Spectrum):
		return x.clip(min, max)

# Numpy universal function for linear interpolation
_ulerp = np.frompyfunc(lerp, 3, 1)


def ufunc_lerp(dt, x, y):
	res = _ulerp(dt, x, y)
	return res.astype(FLOAT)
