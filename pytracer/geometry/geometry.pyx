# distutils: language=c++
# distutils: include_dirs = pytracer/core/
"""
geometry.pyd

pytracer.geometry implementation

Created by Jiayao on Aug 28, 2017
"""
from __future__ import (absolute_import, division)
import cython
from pytracer.core.definition import FLOAT

__all__ = ['Vector', 'Point', 'Normal', 'Ray', 'RayDifferential',
           'BBox']


cdef class _Arr3:
	"""Array baseclass"""

	def __cinit__(self, FLOAT_t x=0., FLOAT_t y=0., FLOAT_t z=0.):
		self.data[0] = x
		self.data[1] = y
		self.data[2] = z

	# fast initialization with __new__
	# def __call__(self, FLOAT_t x=0., FLOAT_t y=0., FLOAT_t z=0.):
	# 	return _Arr3.__new__(x, y, z)

	def __repr__(self):
		return "{}:\n[{}, {}, {}]\n".format(self.__class__, self.data[0], self.data[1], self.data[2])

	@cython.boundscheck(False)
	def __getitem__(self, int item):
		if item == 0:
			return self.data[0]
		elif item == 1:
			return self.data[1]
		elif item == 2:
			return self.data[2]
		raise IndexError

	@cython.boundscheck(False)
	def __setitem__(self, int key, FLOAT_t value):
		if key == 0:
			self.data[0] = value
		elif key == 1:
			self.data[1] = value
		elif key == 2:
			self.data[2] = value
		raise IndexError

	@property
	def x(self):
		return self.data[0]

	@property
	def y(self):
		return self.data[1]

	@property
	def z(self):
		return self.data[2]

	@x.setter
	def x(self, FLOAT_t v):
		self.data[0] = v

	@y.setter
	def y(self, FLOAT_t v):
		self.data[1] = v

	@z.setter
	def z(self, FLOAT_t v):
		self.data[2] = v

	def __add__(self, other):
		cdef FLOAT_t o
		if isinstance(self, _Arr3):
			if isinstance(other, _Arr3):
				return self.__class__(self.x + other.x,
				                      self.y + other.y,
				                      self.z + other.z)
			else:
				o = <FLOAT_t> other
				return self.__class__(self.x + o,
				                      self.y + o,
				                      self.z + o)
		elif isinstance(other, _Arr3):
			# assert type(self) == FLOAT
			return other.__class__(self + other.x,
			                      self + other.y,
			                      self + other.z)
		raise TypeError

	def __neg__(self):
		self.data[0] *= -1
		self.data[1] *= -1
		self.data[2] *= -1
		return self

	def __mul__(self, other):
		cdef FLOAT_t o
		if isinstance(self, _Arr3):
			if isinstance(other, _Arr3):
				return self.__class__(self.x * other.x,
				                      self.y * other.y,
				                      self.z * other.z)
			else:
				o = <FLOAT_t> other
				return self.__class__(self.x * o,
				                      self.y * o,
				                      self.z * o)
		elif isinstance(other, _Arr3):
			# assert type(self) == FLOAT
			return other.__class__(self * other.x,
			                      self * other.y,
			                      self * other.z)

		raise TypeError

	def __truediv__(self, other):
		cdef FLOAT_t o
		if isinstance(self, _Arr3):
			if isinstance(other, _Arr3):
				return self.__class__(self.x / other.x,
				                      self.y / other.y,
				                      self.z / other.z)
			else:
				o = <FLOAT_t> other
				return self.__class__(self.x / o,
				                      self.y / o,
				                      self.z / o)
		elif isinstance(other, _Arr3):
			# assert type(self) == FLOAT
			return other.__class__(self / other.x,
			                      self / other.y,
			                      self / other.z)

		raise TypeError

	def __sub__(self, other):
		cdef FLOAT_t o
		if isinstance(self, _Arr3):
			if isinstance(other, _Arr3):
				return self.__class__(self.x - other.x,
				                      self.y - other.y,
				                      self.z - other.z)
			# elif type(other) == FLOAT:
			else:
				o = <FLOAT_t> other
				return self.__class__(self.x - other,
				                      self.y - other,
				                      self.z - other)
		elif isinstance(other, _Arr3):
			# assert type(self) == FLOAT
			return other.__class__(self - other.x,
			                      self - other.y,
			                      self - other.z)
		raise TypeError

	cpdef FLOAT_t dot(self, _Arr3 other):
		return self._dot(other)

	cpdef _Arr3 cross(self, _Arr3 other):
		return self._cross(other)

	cpdef FLOAT_t sq_length(self):
		return self._sq_length()

	cpdef FLOAT_t length(self):
		return self._length()

	cpdef _Arr3 normalize(self):
		return self._normalize()

cdef class Point(_Arr3):
	def __sub__(self, other):
		cdef FLOAT_t o
		if isinstance(self, Point):
			if isinstance(other, Point):
				return Vector(self.x - other.x,
		                      self.y - other.y,
		                      self.z - other.z)
			if isinstance(other, _Arr3):
				return Point(self.x - other.x,
				             self.y - other.y,
				             self.z - other.z)
			# elif type(other) == FLOAT:
			else:
				o = <FLOAT_t> other
				return self.__class__(self.x - other,
				                      self.y - other,
				                      self.z - other)
		elif isinstance(other, Point):
			# assert type(self) == FLOAT
			return self.__class__(self - other.x,
			                      self - other.y,
			                      self - other.z)

		raise TypeError

	cpdef FLOAT_t sq_dist(self, Point p):
		return self._sq_dist(p)

	cpdef FLOAT_t dist(self, Point p):
		return self._dist(p)

cdef class Ray:
	"""Ray Class"""

	def __cinit__(self, Point o=Point(0., 0., 0.),
	              Vector d=Vector(0., 0., 0.),
	              FLOAT_t mint=0., FLOAT_t maxt=INF, INT_t depth=0, FLOAT_t time=0.):
		self.o = o
		self.d = d
		self.mint = mint
		self.maxt = maxt
		self.depth = depth
		self.time = time

	def __repr__(self):
		return "{}\nOrigin:    {}\nDirection: {}\nmint: {}    \
				maxt: {}\ndepth: {}\ntime: {}\n".format(self.__class__,
		        self.o, self.d, self.mint, self.maxt, self.depth, self.time)


	def __call__(self, FLOAT_t t):
		print("Depreciated: use {}.at() instead".format(self.__class__))
		return self.at(t)

	@staticmethod
	def from_ray(Ray r not None):
		print("Ray.from_ray(): testing only")
		return Ray._from_ray(r)

	@staticmethod
	def from_parent(Point o not None, Vector d not None, Ray r not None,
	                 FLOAT_t mint, FLOAT_t maxt):
		print("Ray.from_parent(): testing only")
		return Ray._from_parent(o, d, r, mint, maxt)

	cpdef Point at(self, FLOAT_t t):
		return self._at(t)

cdef class RayDifferential(Ray):
	def __cinit__(self, Point o=Point(0., 0., 0.),
	              Vector d=Vector(0., 0., 0.), FLOAT_t mint=0., FLOAT_t maxt=0.,
	              INT_t depth=0, FLOAT_t time=0.):
		super().__init__(o, d, mint, maxt, depth, time)
		self.has_differential = 0
		self.rxOrigin = Point(0., 0., 0.)
		self.ryOrigin = Point(0., 0., 0.)
		self.rxDirection = Vector(0., 0., 0.)
		self.ryDirection = Vector(0., 0., 0.)

	@staticmethod
	def from_rd(RayDifferential r not None):
		print("RayDifferential.from_rd(): testing only")
		return RayDifferential._from_rd(r)


cdef class BBox:
	"""BBox Class"""
	def __cinit__(self, Point p1=None, Point p2=None):
		if p1 is not None and p2 is not None:
			self.pMin = Point(fmin(p1.x, p2.x),
			                  fmin(p1.y, p2.y),
			                  fmin(p1.z, p2.z))
			self.pMax = Point(fmax(p1.x, p2.x),
			                  fmax(p1.y, p2.y),
			                  fmax(p1.z, p2.z))

		elif p1 is None and p2 is None:
			self.pMin = Point(INF, INF, INF)
			self.pMax = Point(-INF, -INF, -INF)

		elif p2 is None:
			self.pMin = p1._copy()
			self.pMax = Point(INF, INF, INF)

		else:
			self.pMin = Point(-INF, -INF, -INF)
			self.pMax = p2._copy()


	def __repr__(self):
		return "{}\npMin:{}\npMax:{}".format(self.__class__,
                                 self.pMin,
                                 self.pMax)

	def __getitem__(self, INT_t item):
		if item == 0:
			return self.pMin
		elif item == 1:
			return self.pMax
		raise KeyError


	@staticmethod
	def Union(BBox b1 not None, b2 not None):
		print("BBox.Union(): Depreciated, using Union_b() or Union_p() instead")
		if isinstance(b2, BBox):
			return BBox._Union_b(b1, b2)
		elif isinstance(b2, Point):
			return BBox._Union_p(b1, b2)

		raise TypeError

	@staticmethod
	def union(self, b2 not None):
		print("BBox.Union(): Depreciated, using Union_b() or Union_p() instead")
		if isinstance(b2, BBox):
			return self._union_b(b2)
		elif isinstance(b2, Point):
			return self._union_p(b2)

		raise TypeError

	@cython.boundscheck(False)
	cdef bool _intersect_p(self, Ray r, FLOAT_t *t0, FLOAT_t *t1):
		cdef:
			FLOAT_t s0 = r.mint, s1 = r.maxt
			FLOAT_t tnear, tfar
			Vector t_near = (self.pMin - r.o) / r.d
			Vector t_far = (self.pMax - r.o) / r.d
			INT_t i = 0

		for i in range(3):
			tnear = (self.pMin[i] - r.o[i]) / r.d[i]
			tfar = (self.pMax[i] - r.o[i]) / r.d[i]

			if tnear > tfar:
				fswap(&tnear, &tfar)
			s0 = fmax(s0, tnear)
			s1 = fmin(s1, tfar)
			if s0 > s1:
				return False

		t0[0] = s0
		t1[0] = s1
		return True

	cpdef bool overlaps(self, BBox other):
		return self._overlaps(other)

	cpdef bool inside(self, Point pnt):
		return self._inside(pnt)

	cpdef bool expand(self, FLOAT_t delta):
		return self._expand(delta)

	cpdef FLOAT_t surface_area(self):
		return self._surface_area()

	cpdef FLOAT_t volume(self):
		return self._volume()

	cpdef INT_t maximum_extent(self):
		return self._maximum_extent()

	cpdef Point lerp(self, FLOAT_t tx, FLOAT_t ty, FLOAT_t tz):
		return self._lerp(tx, ty, tz)

	cpdef Vector offset(self, Point pnt):
		return self._offset(pnt)

	def bounding_shpere(self, Point ctr not None):
		cdef FLOAT_t rad = self._bounding_sphere(ctr)
		return [ctr, rad]

	def intersect_p(self, Ray r not None):
		cdef FLOAT_t t0 = 0., t1 = 0.
		if self._intersect_p(r, &t0, &t1):
			return [True, t0, t1]
		return [False, 0., 0.]




