"""
definition.py

pytracer.utility package

Define type definitions with Cython.

Created by Jiayao on Aug 28, 2017
"""
# distutils: language=c++
from __future__ import absolute_import
from libc cimport math
from libc.stdlib cimport malloc, free
cimport scipy.linalg.cython_lapack as cython_lapack
import numpy as np
cimport numpy as np

FLOAT = np.float32
ctypedef np.float32_t FLOAT_t
ctypedef np.float64_t DOUBLE_t
ctypedef np.int32_t INT_t
ctypedef np.uint32_t UINT32_t
ctypedef np.uint8_t UINT8_t

ctypedef FLOAT_t[:, :] mat4x4

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

cdef inline mat4x4 mat4x4_inv(mat4x4 mat):
	cdef int k = 4, zero = 0
	cdef DOUBLE_t *mat_ptr = <DOUBLE_t*> np.PyArray_DATA(mat)
	cdef DOUBLE_t *id_ptr
	cdef int *pvt_ptr = <int*> malloc(sizeof (int) * k)
	cdef DOUBLE_t[:, :] identity = np.eye(k, dtype=np.dtype('float64'))

	try:
		id_ptr = <DOUBLE_t*> np.PyArray_DATA(identity)
		cython_lapack.dgesv(&k, &k, mat_ptr, &k,
		                    pvt_ptr, id_ptr, &k, &zero)
		return identity.astype('float32')

	finally:
		free(pvt_ptr)


cdef inline void mat4x4_kji(mat4x4 m1, mat4x4 m2, mat4x4 res):
	cdef INT_t i, j, k
	cdef FLOAT_t tmp
	for i in range(4):
		for j in range(4):
			res[i][j] = 0.

	for k in range(4):
		for j in range(4):
			tmp = m2[k][j]
			for i in range(4):
				res[i][j] += m1[i][k] * tmp

cdef inline void mat4x4_t(mat4x4 m, mat4x4 m_inv):
	cdef INT_t i, j
	for i in range(4):
		for j in range(4):
			m_inv[i][j] = m[j][i]

cdef inline FLOAT_t det3x3(mat4x4 m):
	return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
	        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
	        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))



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
