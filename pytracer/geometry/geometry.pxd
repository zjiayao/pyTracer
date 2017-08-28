"""
geometry.pyd

pytracer.geometry definition

Created by Jiayao on Aug 28, 2017
"""
# distutils: language=c++
from __future__ import absolute_import
from libcpp cimport bool
from libc cimport math
from pytracer.core.definition cimport FLOAT_t, INT_t, INF, feq


cdef class _Arr3:
	"""Array baseclass"""
	cdef FLOAT_t data[3]

	@staticmethod
	cdef inline _Arr3 from_arr(_Arr3 n):
		return n.__class__(n[0], n[1], n[2])

	cdef inline _Arr3 _copy(self):
		return self.__class__(self.data[0], self.data[1], self.data[2])

	cdef inline FLOAT_t _dot(self, _Arr3 other):
		return self.x * other.x + self.y * other.y + self.z * other.z


	cdef inline _Arr3 _cross(self, _Arr3 other):
		return self.__class__(self.y * other.z - self.z * other.y,
		                              self.z * other.x - self.x * other.z,
		                              self.x * other.y - self.y * other.x)


	cdef inline FLOAT_t _sq_length(self):
		return self.x * self.x + self.y * self.y + self.z * self.z


	cdef inline FLOAT_t _length(self):
		return math.sqrt(self.sq_length())


	cdef inline _Arr3 _normalize(self):
		"""Inplace normalization"""
		cdef FLOAT_t length = self.length()
		if not feq(length, 0.):
			self.data[0] /= length
			self.data[1] /= length
			self.data[2] /= length
		return self

	cpdef FLOAT_t dot(self, _Arr3 other)
	cpdef _Arr3 cross(self, _Arr3 other)
	cpdef FLOAT_t sq_length(self)
	cpdef FLOAT_t length(self)
	cpdef _Arr3 normalize(self)

cdef class Vector(_Arr3):
	pass


cdef class Normal(_Arr3):
	pass


cdef class Point(_Arr3):
	pass


cdef class Ray:
	cdef:
		Point o
		Vector d
		FLOAT_t mint, maxt, time
		INT_t depth

	@staticmethod
	cdef inline Ray _from_parent(Point o, Vector d, Ray r,
	                 FLOAT_t mint, FLOAT_t maxt):
		return r.__class__(o, d, mint, maxt, r.depth + 1, r.time)

	@staticmethod
	cdef inline Ray _from_ray(Ray r):
		return r.__class__(r.o, r.d, r.mint, r.maxt, r.depth, r.time)

	cdef inline Point _at(self, FLOAT_t t):
		return self.o + self.d * t

	cpdef Point at(self, FLOAT_t t)

cdef class RayDifferential(Ray):
	cdef:
		bool has_differentials
		Point rxOrigin, ryOrigin
		Vector rxDirection, ryDirection


	@staticmethod
	cdef inline RayDifferential _from_rd(RayDifferential r):
		"""
		initialize from a `RayDifferential`, analogous to a copy constructor
		"""
		self = RayDifferential(r.o, r.d, r.mint, r.maxt, r.depth, r.time)
		self.has_differentials = r.has_differentials
		self.rxOrigin = r.rxOrigin._copy()
		self.ryOrigin = r.ryOrigin._copy()
		self.rxDirection = r.rxDirection._copy()
		self.ryDirection = r.ryDirection._copy()

		return self

	cdef inline RayDifferential scale_differential(self, FLOAT_t s):
		self.rxOrigin = self.o + (self.rxOrigin - self.o) * s
		self.ryOrigin = self.o + (self.ryOrigin - self.o) * s
		self.rxDirection = self.d + (self.rxDirection - self.d) * s
		self.ryDirection = self.d + (self.ryDirection - self.d) * s
		return self



