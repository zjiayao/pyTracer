"""
geometry.pyd

pytracer.geometry definition

Created by Jiayao on Aug 28, 2017
"""
# distutils: language=c++
from __future__ import absolute_import
from libcpp cimport bool
from libc cimport math
from pytracer.core.definition cimport FLOAT_t, INT_t, INF, feq, fmin, fmax,lerp, fswap


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
	cdef inline FLOAT_t _sq_dist(self, Point p):
		return (p.x - self.x) * (p.x - self.x) + (p.y - self.y) * (p.y - self.y) + (p.z - self.z) * (p.z - self.z)

	cdef inline FLOAT_t _dist(self, Point p):
		return math.sqrt(self._sq_dist(p))

	cpdef FLOAT_t sq_dist(self, Point)
	cpdef FLOAT_t dist(self, Point)


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


cdef class BBox:
	"""BBox Class"""
	cdef:
		public Point pMin, pMax


	@staticmethod
	cdef inline _Arr3 from_bbox(BBox box):
		return BBox.__class__(box.pMin, box.pMax)

	@staticmethod
	cdef inline BBox _Union_b(BBox b1, BBox b2):
		cdef BBox ret = BBox()
		ret.pMin.x = fmin(b1.pMin.x, b2.pMin.x)
		ret.pMin.y = fmin(b1.pMin.y, b2.pMin.y)
		ret.pMin.z = fmin(b1.pMin.z, b2.pMin.z)
		ret.pMax.x = fmax(b1.pMax.x, b2.pMax.x)
		ret.pMax.y = fmax(b1.pMax.y, b2.pMax.y)
		ret.pMax.z = fmax(b1.pMax.z, b2.pMax.z)
		return ret

	@staticmethod
	cdef inline BBox _Union_p(BBox b1, Point b2):
		cdef BBox ret = BBox()
		ret.pMin.x = fmin(b1.pMin.x, b2.x)
		ret.pMin.y = fmin(b1.pMin.y, b2.y)
		ret.pMin.z = fmin(b1.pMin.z, b2.z)
		ret.pMax.x = fmax(b1.pMax.x, b2.x)
		ret.pMax.y = fmax(b1.pMax.y, b2.y)
		ret.pMax.z = fmax(b1.pMax.z, b2.z)
		return ret

	cdef inline BBox _union_b(self, BBox other):
		self.pMin.x = fmin(self.pMin.x, other.pMin.x)
		self.pMin.y = fmin(self.pMin.y, other.pMin.y)
		self.pMin.z = fmin(self.pMin.z, other.pMin.z)
		self.pMax.x = fmax(self.pMax.x, other.pMax.x)
		self.pMax.y = fmax(self.pMax.y, other.pMax.y)
		self.pMax.z = fmax(self.pMax.z, other.pMax.z)
		return self

	cdef inline BBox _union_p(self, Point other):
		self.pMin.x = fmin(self.pMin.x, other.x)
		self.pMin.y = fmin(self.pMin.y, other.y)
		self.pMin.z = fmin(self.pMin.z, other.z)
		self.pMax.x = fmax(self.pMax.x, other.x)
		self.pMax.y = fmax(self.pMax.y, other.y)
		self.pMax.z = fmax(self.pMax.z, other.z)
		return self

	cdef inline bool _overlaps(self, BBox other):
		return (self.pMax.x >= other.pMin.x) and (self.pMin.x <= other.pMax.x) and \
		       (self.pMax.y >= other.pMin.y) and (self.pMin.y <= other.pMax.y) and \
		       (self.pMax.z >= other.pMin.z) and (self.pMin.z <= other.pMax.z)

	cdef inline bool _inside(self, Point pnt):
		return (self.pMax.x >= pnt.x) and (self.pMin.x <= pnt.x) and \
		       (self.pMax.y >= pnt.y) and (self.pMin.y <= pnt.y) and \
		       (self.pMax.z >= pnt.z) and (self.pMin.z <= pnt.z)

	cdef inline BBox _expand(self, FLOAT_t delta):
		self.pMin.x -= delta
		self.pMin.y -= delta
		self.pMin.z -= delta
		self.pMax.x += delta
		self.pMax.y += delta
		self.pMax.z += delta
		return self

	cdef inline FLOAT_t _surface_area(self):
		cdef Vector d = self.pMax - self.pMin
		return 2. * (d.x * d.y + d.x * d.z + d.y * d.z)

	cdef inline FLOAT_t _volume(self):
		cdef Vector d = self.pMax - self.pMin
		return (self.pMax.x - self.pMin.x) * (self.pMax.y - self.pMin.y) * (self.pMax.z - self.pMin.z)

	cdef inline INT_t _maximum_extent(self):
		cdef:
			FLOAT_t dx = self.pMax.x - self.pMin.x
			FLOAT_t	dy = self.pMax.y - self.pMin.y
			FLOAT_t	dz = self.pMax.z - self.pMin.z
		if dx > dy and dx > dz:
			return 0
		elif dy > dz:
			return 1
		return 2

	cdef inline Point _lerp(self, FLOAT_t tx, FLOAT_t ty, FLOAT_t tz):
		return Point(lerp(tx, self.pMin.x, self.pMax.x),
		             lerp(ty, self.pMin.y, self.pMax.y),
		             lerp(tz, self.pMin.z, self.pMax.z))

	cdef inline Vector _offset(self, Point pnt):
		return Vector((pnt.x - self.pMin.x) / (self.pMax.x - self.pMin.x),
		              (pnt.y - self.pMin.y) / (self.pMax.y - self.pMin.y),
		              (pnt.z - self.pMin.z) / (self.pMax.z - self.pMin.z))

	cdef inline FLOAT_t bounding_sphere(self, Point ctr):
		ctr.x = .5 * (self.pMin.x + self.pMax.x)
		ctr.y = .5 * (self.pMin.y + self.pMax.y)
		ctr.z = .5 * (self.pMin.z + self.pMax.z)
		cdef FLOAT_t rad = 0.
		if self._inside(ctr):
			rad = ctr._dist(self.pMax)
		return rad

	cdef bool _intersect_p(self, Ray r, FLOAT_t *t0, FLOAT_t *t1)
	cpdef bool overlaps(self, BBox other)
	cpdef bool inside(self, Point pnt)
	cpdef bool expand(self, FLOAT_t delta)
	cpdef FLOAT_t surface_area(self)
	cpdef FLOAT_t volume(self)
	cpdef INT_t maximum_extent(self)
	cpdef Point lerp(self, FLOAT_t tx, FLOAT_t ty, FLOAT_t tz)
	cpdef Vector offset(self, Point pnt)




