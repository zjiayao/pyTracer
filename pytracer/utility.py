"""
utility.py

pytracer package

Defines global constants and
utility functions.

Created by Jiayao on Aug 13, 2017
"""
from __future__ import absolute_import
import numpy as np
from . import (FLOAT, INT, UINT, EPS)


# Global Functions
def feq(x: FLOAT, y: FLOAT) -> bool:
	"""Equality test for floats"""
	return np.isclose(x, y, atol=EPS)


def eq_unity(x: FLOAT) -> bool:
	"""Equality test with unity"""
	return (x > 1. - EPS) and (x < 1. + EPS)


def ne_unity(x: FLOAT) -> bool:
	"""Inequality test with unity"""
	return x < 1. - EPS or x > 1. + EPS


def ftoi(x: FLOAT) -> INT:
	"""Floor to integer"""
	return INT(np.floor(x))


def ctoi(x: FLOAT) -> INT:
	"""Ceiling to integer"""
	return INT(np.ceil(x))


def rtoi(x: FLOAT) -> INT:
	"""Round to integer"""
	return INT(np.round(x))


def lerp(t: FLOAT, v1: FLOAT, v2: FLOAT) -> FLOAT:
	"""Linear interpolation between `v1` and `v2`"""
	return (1. - t) * v1 + t * v2


def round_pow_2(x: INT) -> UINT:
	"""Round to nearest power of 2"""
	return UINT(2 ** np.round(np.log2(x)))


def next_pow_2(x: INT) -> UINT:
	"""Round to next(or current) power of 2"""
	return UINT(2 ** np.ceil(np.log2(x)))


def is_pow_2(x: INT) -> bool:
	"""Test whether is power of 2"""
	return True if x == 0 else (np.log2(x) % 1) == 0.

# Numpy universal function for linear interpolation
ufunc_lerp = np.frompyfunc(lerp, 3, 1)
