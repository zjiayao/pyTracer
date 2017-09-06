"""
utility.pyx

This module is part of the pyTracer, which
defines geometric inline functions.

v0.0
Created by Jiayao on July 28, 2017
Modified on Aug 13, 2017
Cythonized on Aug 30, 2017
"""
from __future__ import absolute_import
from pytracer.geometry.geometry cimport Vector, Normal
from pytracer.core.definition cimport PI, FLOAT_t, fabs, fclip, fsqrt, fsin, fcos, facos, fatan2

__all__ = ['coordinate_system',
           'face_forward',
           'spherical_direction',
           'spherical_theta',
           'spherical_phi']

# Geometry Utility Functions
cpdef list coordinate_system(Vector v):
	cdef Vector v2 = Vector()
	cdef Vector v3 = Vector()
	_coordinate_system(v, v2, v3)
	return [v, v2, v3]


cdef inline void _coordinate_system(Vector v1, Vector v2, Vector v3):
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
	cdef FLOAT_t inv_len

	if v1.length() == 0.:
		v1._set(1., 0., 0.)
		v2._set(0., 1., 0.)
		v3._set(0., 0., 1.)
		return

	if fabs(v1.x) > fabs(v1.y):
		inv_len = 1. / fsqrt(v1.x * v1.x + v1.z * v1.z)
		v2._set(-v1.z * inv_len, 0., v1.x * inv_len)

	else:
		inv_len = 1. / fsqrt(v1.y * v1.y + v1.z * v1.z)
		v2._set(0., -v1.z * inv_len, v1.y * inv_len)

	v3._set_cross(v1, v2)

cpdef Normal face_forward(Normal n, Vector v):
	return _face_forward(n, v)

cdef inline Normal _face_forward(Normal n, Vector v):
	"""
	face_forward

	Flip a `Normal` according to a `Vector` such that
	they make an angle less than pi
	"""
	if not isinstance(n, Normal) or not isinstance(v, Vector):
		raise TypeError('Argument type error')

	if n.dot(v) < 0.:
		return -n.copy()

	return n.copy()

cpdef Vector spherical_direction(FLOAT_t stheta, FLOAT_t ctheta, FLOAT_t phi, Vector x=None, Vector y=None, Vector z=None):
	return _spherical_direction(stheta, ctheta, phi, x, y, z)

cdef inline Vector _spherical_direction(FLOAT_t stheta, FLOAT_t ctheta, FLOAT_t phi, Vector x=None, Vector y=None, Vector z=None):
	"""
	spherical_direction

	Computes spherical direction from sperical coordiante
	with or without basis.
	"""
	if x is None or y is None or z is None:
		return Vector(stheta * fcos(phi), stheta * fsin(phi), ctheta)
	else:
		return x * stheta * fcos(phi) + y * stheta * fsin(phi) + z * ctheta

cpdef FLOAT_t spherical_theta(Vector v):
	return _spherical_theta(v)

cdef inline FLOAT_t _spherical_theta(Vector v):
	"""
	spherical_theta

	Get theta from direction
	"""
	return facos(fclip(v.z, -1., 1.))

cpdef FLOAT_t spherical_phi(Vector v):
	return _spherical_phi(v)

cdef inline FLOAT_t _spherical_phi(Vector v):
	"""
	spherical_phi

	Get phi from direction
	"""
	cdef FLOAT_t p = fatan2(v.y, v.x)
	return p if p > 0. else (p + 2. * PI)

