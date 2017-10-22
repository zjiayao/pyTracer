"""
definition.py

pytracer.utility package

Define type definitions with Cython.

Created by Jiayao on Aug 28, 2017
"""
# distutils: language=c++
from __future__ import absolute_import
from libc cimport math
# from libc.stdlib cimport malloc, free
# cimport scipy.linalg.cython_lapack as cython_lapack
import numpy as np
cimport numpy as np

FLOAT = np.float32
ctypedef np.float32_t FLOAT_t
# ctypedef np.float64_t DOUBLE_t
ctypedef np.int32_t INT_t
ctypedef np.uint32_t UINT_t
ctypedef np.uint8_t UINT8_t


cdef extern from 'constant.hpp':
	FLOAT_t EPS, PI, INV_PI, INV_2PI, INF

cdef inline FLOAT_t fsqrt(FLOAT_t x):
	return math.sqrt(x)

cdef inline FLOAT_t fsin(FLOAT_t x):
	return math.sin(x)

cdef inline FLOAT_t fasin(FLOAT_t x):
	return math.asin(x)

cdef inline FLOAT_t fcos(FLOAT_t x):
	return math.cos(x)

cdef inline FLOAT_t facos(FLOAT_t x):
	return math.acos(x)

cdef inline FLOAT_t ftan(FLOAT_t x):
	return math.tan(x)

cdef inline FLOAT_t fatan(FLOAT_t x):
	return math.atan(x)

cdef inline FLOAT_t fatan2(FLOAT_t y, FLOAT_t x):
	return math.atan2(y, x)

cdef inline FLOAT_t fabs(FLOAT_t x):
	return x if x >= 0. else -x

cdef inline bint is_inf(FLOAT_t x):
	return x == INF

cdef inline void fswap(FLOAT_t *x, FLOAT_t *y):
	cdef FLOAT_t tmp = x[0]
	x[0] = y[0]
	y[0] = x[0]

cdef inline FLOAT_t fmin(FLOAT_t x, FLOAT_t y):
	return x if x <= y else y

cdef inline FLOAT_t fmax(FLOAT_t x, FLOAT_t y):
	return x if x > y else y

cdef inline bint feq(FLOAT_t x, FLOAT_t y):
	return y - x < EPS and x - y < EPS

cdef inline bint eq_unity(FLOAT_t x):
	return feq(x, 1.)

cdef inline bint ne_unity(FLOAT_t x):
	return not feq(x, 1.)

cdef inline bint is_zero(FLOAT_t x):
	return x > -EPS and x < EPS

cdef inline bint not_zero(FLOAT_t x):
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

cdef inline bint is_pow_2(INT_t x):
	return x & (x - 1) == 0

cdef inline FLOAT_t fclip(FLOAT_t x, FLOAT_t lo, FLOAT_t hi):
	if x <= lo:
		return lo
	elif x >= hi:
		return hi
	return x

cdef inline FLOAT_t deg2rad(FLOAT_t angle):
	return angle / 180. * PI

# cdef inline FLOAT_t fclip(FLOAT_t x):
# 	if x <= 0.:
# 		return 0.
# 	return x

cdef inline bint solve_linear_2x2(const FLOAT_t A[2][2], const FLOAT_t B[2],
                              FLOAT_t *x0, FLOAT_t *x1):
	cdef FLOAT_t det = A[0][0] * A[1][1] - A[0][1] - A[1][0]
	if fabs(det) < EPS:
		return 0
	x0[0] = (A[1][1] * B[0] - A[0][1] * B[1]) / det
	x1[0] = (A[0][0] * B[1] - A[1][0] * B[0]) / det

	if is_inf(x0[0]) or is_inf(x1[0]):
		return 0

	return 1
