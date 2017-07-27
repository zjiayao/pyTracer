'''
pytracer.py

This module is part of the pyTracer, which
holds global definitions

Created by Jiayao on July 27, 2017
'''

from __future__ import print_function
import numpy as np

EPS = 1e-5
INT = np.int32
FLOAT = np.float32
DOUBLE = np.float64
HANDNESS = 'left'

# Forward class declarations

class Vector():
	pass

class Point():
	pass

class Normal():
	pass

class Ray():
	pass

class RayDifferential():
	pass

class BBox():
	pass

class Transform():
	pass


# Global Functions

def feq(x: FLOAT, y: FLOAT):
	return np.isclose(x, y, atol=EPS)

def eq_unity(x: FLOAT):
	return x > 1. - EPS and x < 1. + EPS	

def ne_unity(x: FLOAT):
	return x < 1. - EPS or x > 1. + EPS	

def Lerp(t: FLOAT, v1: FLOAT, v2: FLOAT):
	'''
	Lerp
	Linear interpolation between `v1` and `v2`
	'''
	return (1. - t) * v1 + t * v2