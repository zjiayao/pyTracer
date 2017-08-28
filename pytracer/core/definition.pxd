"""
definition.py

pytracer.utility package

Define type definitions with Cython.

Created by Jiayao on Aug 28, 2017
"""
# distutils: language=c++
from __future__ import absolute_import
from libcpp cimport bool
cimport numpy as np

ctypedef np.float32_t FLOAT_t
ctypedef np.int32_t INT_t
ctypedef np.uint32_t UINT32_t
ctypedef np.uint8_t UINT8_t

from libc cimport math

cdef:
	FLOAT_t EPS = 1e-5
	FLOAT_t PI =  3.14159265358979323846
	FLOAT_t INV_PI = 0.31830988618379067154
	FLOAT_t INV_2PI = 0.15915494309189533577
	FLOAT_t INF = math.INFINITY


cdef inline FLOAT_t fmin(FLOAT_t x, FLOAT_t y):
	return x if x <= y else y

cdef inline FLOAT_t fmax(FLOAT_t x, FLOAT_t y):
	return x if x > y else y

cdef inline bool feq(FLOAT_t x, FLOAT_t y):
	return x - y > -EPS and y - x < EPS

cdef inline bool eq_unity(FLOAT_t x):
	return feq(x, 1.)

cdef inline bool ne_unity(FLOAT_t x):
	return not feq(x, 1.)

cdef inline bool is_zero(FLOAT_t x):
	return x > -EPS and x < EPS

cdef inline bool not_zero(FLOAT_t x):
	return x > EPS or x < -EPS


cdef inline INT_t ftoi(FLOAT_t x):
	return <INT_t> math.floor(x)

cdef inline INT_t ctoi(FLOAT_t x):
	return <INT_t> math.ceil(x)

cdef inline INT_t rtoi(FLOAT_t x):
	return <INT_t> math.round(x)

cdef inline FLOAT_t lerp(FLOAT_t t, FLOAT_t v1, FLOAT_t v2):
	return (1. - t) * v1 + t * v2

cdef inline INT_t round_pow_2(INT_t x) except -1:
	"""INT_t is np.int32_t"""
	x -= 1
	x |= x >> 1
	x |= x >> 2
	x |= x >> 4
	x |= x >> 8
	x |= x >> 16
	return x + 1

cdef inline INT_t next_pow_2(INT_t x) except -1:
	return round_pow_2(x)

cdef inline bool is_pow_2(INT_t x):
	return x & (x - 1) == 0

cdef inline FLOAT_t clip(FLOAT_t x, FLOAT_t lo, FLOAT_t hi):
	if x <= lo:
		return lo
	elif x >= hi:
		return hi
	return x

cdef inline FLOAT_t fclip(FLOAT_t x):
	if x <= 0.:
		return 0.
	elif x >= INF:
		return INF
	return x