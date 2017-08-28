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
		if isinstance(self, _Arr3):
			if isinstance(other, _Arr3):
				return self.__class__(self.x + other.x,
				                      self.y + other.y,
				                      self.z + other.z)
			elif type(other) == FLOAT:
				return self.__class__(self.x + other,
				                      self.y + other,
				                      self.z + other)
		elif type(other) == _Arr3:
			assert type(self) == FLOAT
			return other.__class__(self + other.x,
			                      self + other.y,
			                      self + other.z)
		raise TypeError

	def __mul__(self, other):
		if isinstance(self, _Arr3):
			if isinstance(other, _Arr3):
				return self.__class__(self.x * other.x,
				                      self.y * other.y,
				                      self.z * other.z)
			elif type(other) == FLOAT:
				return self.__class__(self.x * other,
				                      self.y * other,
				                      self.z * other)
		elif type(other) == _Arr3:
			assert type(self) == FLOAT
			return other.__class__(self * other.x,
			                      self * other.y,
			                      self * other.z)

		raise TypeError

	def __truediv__(self, other):
		if isinstance(self, _Arr3):
			if isinstance(other, _Arr3):
				return self.__class__(self.x / other.x,
				                      self.y / other.y,
				                      self.z / other.z)
			elif type(other) == FLOAT:
				return self.__class__(self.x / other,
				                      self.y / other,
				                      self.z / other)
		elif type(other) == _Arr3:
			assert type(self) == FLOAT
			return other.__class__(self / other.x,
			                      self / other.y,
			                      self / other.z)

		raise TypeError

	def __sub__(self, other):
		if isinstance(self, _Arr3):
			if isinstance(other, _Arr3):
				return self.__class__(self.x - other.x,
				                      self.y - other.y,
				                      self.z - other.z)
			elif type(other) == FLOAT:
				return self.__class__(self.x - other,
				                      self.y - other,
				                      self.z - other)
		elif type(other) == _Arr3:
			assert type(self) == FLOAT
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
		if isinstance(self, Point):
			if isinstance(other, Point):
				return Vector(self.x - other.x,
		                      self.y - other.y,
		                      self.z - other.z)
			if isinstance(other, _Arr3):
				return Point(self.x - other.x,
				             self.y - other.y,
				             self.z - other.z)
			elif type(other) == FLOAT:
				return self.__class__(self.x - other,
				                      self.y - other,
				                      self.z - other)
		elif type(other) == Point:
			assert type(self) == FLOAT
			return self.__class__(self - other.x,
			                      self - other.y,
			                      self - other.z)

		raise TypeError

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
	def from_ray(Ray r):
		print("Ray.from_ray(): testing only")
		return Ray._from_ray(r)

	@staticmethod
	def from_parent(Point o, Vector d, Ray r,
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
	def from_rd(RayDifferential r):
		print("RayDifferential.from_rd(): testing only")
		return RayDifferential._from_rd(r)

