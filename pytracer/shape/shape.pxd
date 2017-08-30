"""
shape.pxd

pytracer.shape package

Contains the base interface for shapes.

Implementation includes:
	- LoopSubdiv
	- TriangleMesh
	- Sphere
	- Cylinder
	- Disk

Created by Jiayao on July 27, 2017
Modified on Aug 13, 2017
Cythonized on Aug 30, 2017
"""
from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
from libcpp.vector cimport vector as cppvector
from pytracer.core.definition cimport EPS, INT_t, FLOAT_t
from pytracer.geometry.geometry cimport Ray, BBox, Point, Normal, Vector
from pytracer.geometry.diffgeom cimport DifferentialGeometry
from pytracer.transform.transform cimport Transform


cdef class Shape(metaclass=ABCMeta):
	"""
	Shape Class

	Base class of shapes.
	"""
	cdef:
		INT_t shape_id
		Transform o2w, w2o
		bint ro, transform_swaps_handedness

	@staticmethod
	cdef inline void _incr_shape_id():
		global next_shape_id
		next_shape_id += 1

	cdef inline BBox _object_bound(self):
		raise NotImplementedError('unimplemented Shape.object_bound() method called')

	cdef inline BBox _world_bound(self):
		return self.o2w(self.object_bound())

	cdef inline bint _can_intersect(self):
		return True

	cdef inline void _refine(self, cppvector[Shape]* refined):
		"""
		If `Shape` cannot intersect,
		return a refined subset
		"""
		raise NotImplementedError('Intersecable shapes are not refineable')

	cdef inline bint _intersect(self, Ray ray, FLOAT_t *thit, FLOAT_t *r_eps, DifferentialGeometry *dg):
		raise NotImplementedError('unimplemented Shape.intersect() method called')

	cdef inline bint _intersect_p(self, Ray ray):
		raise NotImplementedError('unimplemented {}.intersect_p() method called'
		                          .format(self.__class__))

	cdef inline void _get_shading_geometry(self, Transform o2w, const DifferentialGeometry *dg, DifferentialGeometry *dgs):
		pass

	cdef inline FLOAT_t _area(self):
		raise NotImplementedError('unimplemented Shape.area() method called')

	cdef inline void _sample(self, FLOAT_t u1, FLOAT_t u2, Point p, Normal n):
		"""
		_sample()

		Returns the position and
		surface normal randomly chosen
		"""
		raise NotImplementedError('unimplemented {}.sample() method called'
		                          .format(self.__class__))

	cdef inline void _sample_p(self, Point pnt, FLOAT_t u1, FLOAT_t u2, Point p, Normal n):
		"""
		_sample_p()

		Returns the position and
		surface normal randomly chosen
		s.t. visible to `pnt`
		"""
		return self._sample(u1, u2, p, n)


	cdef inline FLOAT_t _pdf(self, Point pnt):
		"""
		_pdf()

		Return the sampling pdf
		"""
		return 1. / self._area()

	cdef inline FLOAT_t _pdf_p(self, Point pnt, Vector wi):
		"""
		_pdf_p()

		Return the sampling pdf w.r.t.
		the _sample_p() method.
		Transforms the density from one
		defined over area to one defined
		over solid angle from `pnt`.
		"""
		# intersect sample ray with area light
		cdef:
			Ray ray = Ray(pnt, wi, EPS)
			FLOAT_t thit, r_eps
			DifferentialGeometry dg_light

		if not self._intersect(ray, &thit, &r_eps, &dg_light):
			return 0.

		# convert light sample weight to solid angle measure
		return (pnt - ray(thit))._sq_length() / (dg_light.nn._abs_dot(-wi) * self._area())


	cpdef BBox object_bound(self)
	cpdef BBox world_bound(self)
	cpdef bint can_intersect(self)
	cpdef void refine(self, cppvector[Shape]* refined)
	cpdef intersect(self, Ray ray)
	cpdef bint intersect_p(self, Ray ray)
	cpdef FLOAT_t area(self)
	cpdef sample(self, FLOAT_t u1, FLOAT_t u2)
	cpdef sample_p(self, Point pnt, FLOAT_t u1, FLOAT_t u2)
	cpdef FLOAT_t pdf(self, Point pnt)
	cpdef FLOAT_t pdf_p(self, Point pnt, Vector wi)








