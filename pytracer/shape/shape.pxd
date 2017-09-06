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
from pytracer.core.definition cimport EPS, INT_t, FLOAT_t
from pytracer.geometry.geometry cimport Ray, BBox, Point, Normal, Vector
cdef class Shape
from pytracer.geometry.diffgeom cimport DifferentialGeometry
from pytracer.transform.transform cimport Transform


cdef class Shape:
	"""
	Shape Class

	Base class of shapes.
	"""
	cdef:
		INT_t shape_id
		Transform o2w, w2o
		bint ro, transform_swaps_handedness

	@staticmethod
	cdef inline void _incr_shape_id()
	cdef inline BBox _object_bound(self)
	cdef inline BBox _world_bound(self)
	cdef inline bint _can_intersect(self)
	cdef inline void _refine(self, list refined)
	cdef inline bint _intersect(self, Ray ray, FLOAT_t *thit, FLOAT_t *r_eps, DifferentialGeometry dg)
	cdef inline bint _intersect_p(self, Ray ray)
	cdef inline void _get_shading_geometry(self, Transform o2w, DifferentialGeometry dg, DifferentialGeometry dgs)
	cdef FLOAT_t _area(self)
	cdef inline void _sample(self, FLOAT_t u1, FLOAT_t u2, Point p, Normal n)
	cdef inline void _sample_p(self, Point pnt, FLOAT_t u1, FLOAT_t u2, Point p, Normal n)
	cdef inline FLOAT_t _pdf(self, Point pnt)
	cdef inline FLOAT_t _pdf_p(self, Point pnt, Vector wi)

	cpdef BBox object_bound(self)
	cpdef BBox world_bound(self)
	cpdef bint can_intersect(self)
	cpdef void refine(self, list refined)
	cpdef intersect(self, Ray ray)
	cpdef bint intersect_p(self, Ray ray)
	cpdef FLOAT_t area(self)
	cpdef list sample(self, FLOAT_t u1, FLOAT_t u2)
	cpdef list sample_p(self, Point pnt, FLOAT_t u1, FLOAT_t u2)
	cpdef FLOAT_t pdf(self, Point pnt)
	cpdef FLOAT_t pdf_p(self, Point pnt, Vector wi)








