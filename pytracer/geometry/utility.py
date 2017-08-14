"""
util.py

This module is part of the pyTracer, which
defines differential geometric operations.

v0.0
Created by Jiayao on July 28, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer import *
from pytracer.geometry import (Vector, Normal)

__all__ = ['coordinate_system',
           'normalize',
           'face_forward',
           'spherical_direction',
           'spherical_theta',
           'spherical_phi']

# Geometry Utility Functions


def coordinate_system(v1: 'Vector') -> ['Vector']:
	"""
	Construct a left-handed coordinate system
	with v1, which is assumed to be normalized.
	Noted left-handed coordinate system is used.

	:param
		- v1: `Vector`
			a `Vector` from which a coordinate
			system is constructed
	:return
		- [`Vector`]
			a python List encompassing three basis
			constructed from `v1`
	"""
	if v1.length() == 0.:
		return [Vector(1., 0., 0.), Vector(0., 1., 0.), Vector(0., 0., 1.)]
	if np.fabs(v1.x) > np.fabs(v1.y):
		inv_len = 1. / np.sqrt(v1.x * v1.x + v1.z * v1.z)
		v2 = Vector(-v1.z * inv_len, 0., v1.x * inv_len)

	else:
		inv_len = 1. / np.sqrt(v1.y * v1.y + v1.z * v1.z)
		v2 = Vector(0., -v1.z * inv_len, v1.y * inv_len)

	v3 = v1.cross(v2)

	return v1, v2, v3


def normalize(vec: 'Vector') -> 'Vector':
	"""
	Normalize a given vector, returns the
	normalized version. The input is not
	modified. Noted `Normal`s are also
	normalized using this method.

	:param
		- vec: `Vector`
			The `Vector` to be normalized.
	:return
		- `Vector`
			The normalized version of `vec`
	"""
	n = vec.copy()
	length = vec.length()
	if not length == 0.:
		n /= length
	return n


def face_forward(n: 'Normal', v: 'Vector') -> 'Normal':
	"""
	face_forward

	Flip a `Normal` according to a `Vector` such that
	they make an angle less than pi
	"""
	if not isinstance(n, Normal) or not isinstance(v, Vector):
		raise TypeError('Argument type error')

	if np.dot(n, v) < 0.:
		return -n.copy()

	return n.copy()


def spherical_direction(stheta: FLOAT, ctheta: FLOAT, phi: FLOAT,
		x: 'Vector'=None, y: 'Vector'=None, z: 'Vector'=None) -> 'Vector':
	"""
	spherical_direction

	Computes spherical direction from sperical coordiante
	with or without basis.
	"""
	if x is None or y is None or z is None:
		return Vector(stheta * np.cos(phi), stheta * np.sin(phi), ctheta)
	else:
		return x * stheta * np.cos(phi) + y * stheta * np.sin(phi) + z * ctheta


def spherical_theta(v: 'Vector') -> FLOAT:
	"""
	spherical_theta

	Get theta from direction
	"""
	return np.arccos(np.clip(v.z, -1., 1.))


def spherical_phi(v: 'Vector') -> FLOAT:
	"""
	spherical_phi

	Get phi from direction
	"""
	p = np.arctan2(v.y, v.x)
	return p if p > 0. else p + 2 * np.pi
