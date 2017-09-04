"""
transform.py

pytracer.transform package

Created by Jiayao on Aug 13, 2017
Cythonized on Aug 30, 2017
"""
from __future__ import absolute_import

from pytracer.core.definition cimport FLOAT_t, INT_t, EPS, mat4x4, feq, fsin, fcos, ftan, deg2rad, is_zero, ne_unity, det3x3, mat4x4_inv, mat4x4_t, mat4x4_kji
from pytracer.geometry.geometry cimport _normalize, _Arr3, Point, Vector, Normal, Ray, RayDifferential, BBox
from cpython.object cimport Py_EQ, Py_NE
cimport numpy as np
cimport cython


cdef class Transform:
	"""
	Transform class
	"""
	cdef:
		mat4x4 m
		mat4x4 m_inv

	cdef inline void _copy(self, Transform to):
		to.m[:, :] = self.m
		to.m_inv[:, :] = self.m_inv

	cdef inline void _call_point(self, Point p, Point to):
		to.data[0] = (self.m[0][0] + self.m[1][0] + self.m[2][0]) * p.data[0] + self.m[0][3]
		to.data[1] = (self.m[0][1] + self.m[1][1] + self.m[2][1]) * p.data[1] + self.m[1][3]
		to.data[2] = (self.m[0][2] + self.m[1][2] + self.m[2][2]) * p.data[2] + self.m[2][3]

		if not feq(to.data[2], 1.):
			to.data[0] /= to.data[2]
			to.data[1] /= to.data[2]
			to.data[2] = 1.

	cdef inline void _call_vector(self, Vector v, Vector to):
		to.data[0] = (self.m[0][0] + self.m[1][0] + self.m[2][0]) * v.data[0]
		to.data[1] = (self.m[0][1] + self.m[1][1] + self.m[2][1]) * v.data[1]
		to.data[2] = (self.m[0][2] + self.m[1][2] + self.m[2][2]) * v.data[2]

	cdef inline void _call_normal(self, Normal n, Normal to):
		to.data[0] = (self.m_inv[0][0] + self.m_inv[0][1] + self.m_inv[0][2]) * n.data[0]
		to.data[1] = (self.m_inv[1][0] + self.m_inv[1][1] + self.m_inv[1][2]) * n.data[1]
		to.data[2] = (self.m_inv[2][0] + self.m_inv[2][1] + self.m_inv[2][2]) * n.data[2]

	cdef inline void _call_ray(self, Ray r, Ray to):
		self._call_point(r.o, to.o)
		self._call_vector(r.d, to.d)

	cdef inline void _call_bbox(self, BBox box, BBox to):
		cdef:
			FLOAT_t x = box.pMax.data[0] - box.pMin.data[0]
			FLOAT_t y = box.pMax.data[1] - box.pMin.data[1]
			FLOAT_t z = box.pMax.data[2] - box.pMin.data[2]
			FLOAT_t xx = (self.m[0][0] + self.m[1][0] + self.m[2][0]) * x
			FLOAT_t yy = (self.m[0][1] + self.m[1][1] + self.m[2][1]) * y
			FLOAT_t zz = (self.m[0][2] + self.m[1][2] + self.m[2][2]) * z

		to.pMin.data[:] = box.pMin.data
		to.pMax.data[0] = to.pMin.data[0] + xx
		to.pMax.data[1] = to.pMin.data[1] + yy
		to.pMax.data[2] = to.pMin.data[2] + zz

	@staticmethod
	cdef inline void _translate(Transform trans, Vector delta):
		trans.m[0][3] = delta.x
		trans.m[1][3] = delta.y
		trans.m[2][3] = delta.z
		trans.m_inv[0][3] = -delta.x
		trans.m_inv[1][3] = -delta.y
		trans.m_inv[2][3] = -delta.z

	@staticmethod
	cdef inline void _scale(Transform trans, FLOAT_t x, FLOAT_t y, FLOAT_t z):
		trans.m[0][0] = x
		trans.m[1][1] = y
		trans.m[2][2] = z
		trans.m_inv[0][0] = 1. / x
		trans.m_inv[1][1] = 1. / y
		trans.m_inv[2][2] = 1. / z

	# all angles are in degrees
	@staticmethod
	cdef inline void _rotate_x(Transform trans, FLOAT_t angle):
		cdef FLOAT_t rad = deg2rad(angle)
		cdef FLOAT_t s = fsin(rad)
		cdef FLOAT_t c = fcos(rad)

		trans.m[1][1] = c
		trans.m[1][2] = -s
		trans.m[2][1] = s
		trans.m[2][2] = c

		trans.m_inv[1][1] = c
		trans.m_inv[1][2] = s
		trans.m_inv[2][1] = -s
		trans.m_inv[2][2] = c

	@staticmethod
	cdef inline void _rotate_y(Transform trans, FLOAT_t angle):
		cdef FLOAT_t rad = deg2rad(angle)
		cdef FLOAT_t s = fsin(rad)
		cdef FLOAT_t c = fcos(rad)

		trans.m[0][0] = c
		trans.m[0][2] = -s
		trans.m[2][0] = s
		trans.m[2][2] = c

		trans.m_inv[0][0] = c
		trans.m_inv[0][2] = s
		trans.m_inv[2][0] = -s
		trans.m_inv[2][2] = c

	@staticmethod
	cdef inline void _rotate_z(Transform trans, FLOAT_t angle):
		cdef FLOAT_t rad = deg2rad(angle)
		cdef FLOAT_t s = fsin(rad)
		cdef FLOAT_t c = fcos(rad)

		trans.m[0][0] = c
		trans.m[0][1] = -s
		trans.m[1][0] = s
		trans.m[1][1] = c

		trans.m_inv[0][0] = c
		trans.m_inv[0][1] = s
		trans.m_inv[1][0] = -s
		trans.m_inv[1][1] = c

	@staticmethod
	cdef inline void _rotate(Transform trans, FLOAT_t angle, Vector axis):
		cdef:
			Vector a = _normalize(axis)
			FLOAT_t rad = deg2rad(angle)
			FLOAT_t s = fsin(rad)
			FLOAT_t c = fcos(rad)

		trans.m[0][0] = a.x * a.x + (1. - a.x * a.x) * c
		trans.m[0][1] = a.x * a.y * (1. - c) - a.z * s
		trans.m[0][2] = a.x * a.z * (1. - c) + a.y * s
		trans.m[1][0] = a.x * a.y * (1. - c) + a.z * s
		trans.m[1][1] = a.y * a.y + (1. - a.y * a.y) * c
		trans.m[1][2] = a.y * a.z * (1. - c) - a.x * s
		trans.m[2][0] = a.x * a.z * (1. - c) - a.y * s
		trans.m[2][1] = a.y * a.z * (1. - c) + a.x * s
		trans.m[2][2] = a.z * a.z + (1. - a.z * a.z) * c
		mat4x4_t(trans.m, trans.m_inv)

	@staticmethod
	cdef void _look_at(Transform trans, Point pos, Point look, Vector up)


	cdef inline Transform _inverse(self):
		"""
		Returns the inverse transformation
		"""
		return Transform(self.m_inv, self.m)

	cdef inline bint _is_identity(self):
		cdef INT_t i, j
		for i in range(4):
			for j in range(4):
				if i == j:
					if not feq(self.m[i][j], 1.) or not feq(self.m_inv[i][j], 1.):
						return False
				else:
					if not is_zero(self.m[i][j]) or not is_zero(self.m_inv[i][j]):
						return False
		return True

	cdef inline bint _has_scale(self):
		cdef INT_t i, j
		cdef FLOAT_t s1, s2
		for i in range(3):
			s1 = 0.
			s2= 0.

			for j in range(3):
				s1 += self.m[i][j] * self.m[i][j]
				s2 += self.m_inv[i][j] * self.m_inv[i][j]

			if ne_unity(s1) or ne_unity(s2):
				return True

		return True

	cdef inline bint _swaps_handedness(self):
		return det3x3(self.m) < 0.

	cpdef Transform copy(self)
	cpdef Transform inverse(self)
	cpdef bint is_identity(self)
	cpdef bint has_scale(self)
	cpdef bint swaps_handedness(self)
#
# cdef class AnimatedTransform:
# 	cdef:
# 		FLOAT_t startTime, endTime
# 		Transform startTransform, endTransform
# 		bint animated
#
# 	def __init__(self, t1: 'Transform', tm1: FLOAT, t2: 'Transform', tm2: FLOAT):
# 		self.startTime = tm1
# 		self.endTime = tm2
# 		self.startTransform = t1
# 		self.endTransform = t2
# 		self.animated = (t1 != t2)
# 		self.T = [None, None]
# 		self.R = [None, None]
# 		self.S = [None, None]
# 		self.T[0], self.R[0], self.S[0] = AnimatedTransform.decompose(t1.m)
# 		self.T[1], self.R[1], self.S[1] = AnimatedTransform.decompose(t2.m)
#
# 	def __repr__(self):
# 		return "{}\nTime: {} - {}\nAnimated: {}".format(self.__class__,
# 		                                                self.startTime, self.endTime, self.animated)
#
# 	def __call__(self, arg_1, arg_2=None):
# 		if isinstance(arg_1, geo.Ray) and arg_2 is None:
# 			r = arg_1
# 			if not self.animated or r.time < self.startTime:
# 				tr = self.startTransform(r)
# 			elif r.time >= self.endTime:
# 				tr = self.endTransform(r)
# 			else:
# 				tr = self.interpolate(r.time)(r)
# 			tr.time = r.time
# 			return tr
#
# 		elif isinstance(arg_1, (float, FLOAT, np.float)) and isinstance(arg_2, geo.Point):
# 			time = arg_1
# 			p = arg_2
# 			if not self.animated or time < self.startTime:
# 				return self.startTransform(p)
# 			elif time > self.endTime:
# 				return self.endTransform(p)
# 			return self.interpolate(time)(p)
#
# 		elif isinstance(arg_1, (float, FLOAT, np.float)) and isinstance(arg_2, geo.Vector):
# 			time = arg_1
# 			v = arg_2
# 			if not self.animated or time < self.startTime:
# 				return self.startTransform(v)
# 			elif time > self.endTime:
# 				return self.endTransform(v)
# 			return self.interpolate(time)(v)
# 		else:
# 			raise TypeError()
#
# 	def motion_bounds(self, b: 'geo.BBox', use_inv: bool=False) -> 'geo.BBox':
# 		if not self.animated:
# 			return self.startTransform.inverse()(b)
# 		ret = geo.BBox()
# 		steps = 128
# 		for i in range(128):
# 			time = util.lerp(i * (1. / (steps - 1)), self.startTime, self.endTime)
# 			t = self.interpolate(time)
# 			if use_inv:
# 				t = t.inverse()
# 			ret.union(t(b))
# 		return ret
#
# 	@staticmethod
# 	def decompose(m: 'np.ndarray') -> ['geo.Vector', 'quat.Quaternion', 'np.ndarray']:
# 		"""
# 		decompose into
# 		m = T R S
# 		Assume m is an affine transformation
# 		"""
# 		if not np.shape(m) == (4, 4):
# 			raise TypeError
#
# 		T = geo.Vector(m[0, 3], m[1, 3], m[2, 3])
# 		M = m.copy()
# 		M[0:3, 3] = M[3, 0:3] = 0
# 		M[3, 3] = 1
#
# 		# polar decomposition
# 		norm = 2 * EPS
# 		R = M.copy()
#
# 		for _ in range(100):
# 			if norm < EPS:
# 				break
# 			Rit = np.linalg.inv(R.T)
# 			Rnext = .5 * (Rit + R)
# 			D = np.fabs(Rnext - Rit)[0:3, 0:3]
# 			norm = max(norm, np.max(np.sum(D, axis=0)))
# 			R = Rnext
#
# 		from pytracer.transform.quat import from_arr
# 		Rquat = from_arr(R)
# 		S = np.linalg.inv(R).dot(M)
#
# 		return T, Rquat, S
#
# 	def interpolate(self, time: FLOAT) -> 'Transform':
#
# 		if not self.animated or time <= self.startTime:
# 			return self.startTransform
#
# 		if time >= self.endTime:
# 			return self.endTransform
#
# 		from pytracer.transform.quat import (slerp, to_transform)
#
# 		dt = (time - self.startTime) / (self.endTime - self.startTime)
#
# 		trans = (1. - dt) * self.T[0] + dt * self.T[1]
# 		rot = slerp(dt, self.R[0], self.R[1])
# 		scale = util.ufunc_lerp(dt, self.S[0], self.S[1])
#
# 		return Transform.translate(trans) *\
# 		       to_transform(rot) *\
# 		       Transform(scale)
