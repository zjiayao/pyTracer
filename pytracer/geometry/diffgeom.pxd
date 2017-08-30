"""
diffgeo.pxd

This module is part of the pyTracer, which
defines differential geometric operations.

v0.0
Created by Jiayao on July 28, 2017
Modified on Aug 13, 2017
Cythonized on Aug 30, 2017
"""
from __future__ import absolute_import
from pytracer.core.definition cimport INT_t, FLOAT_t, fabs, is_zero, solve_linear_2x2
from pytracer.geometry.geometry cimport Point, Vector, Normal, RayDifferential
from pytracer.shape.shape cimport Shape

__all__ = ['DifferentialGeometry']

cdef struct DifferentialGeometry:
	Point p
	Vector dpdu, dpdv, dpdx, dpdy
	Normal dndu, dndv, nn
	FLOAT_t u, v, dudx, dudy, dvdx, dvdy
	Shape shape
	
cdef inline void _diffgeom_init(DifferentialGeometry *self, Point p, Vector dpdu, Vector dpdv, Normal dndu,
                                Normal dndv, FLOAT_t uu, FLOAT_t vv, Shape shape):
		self.p = p.copy()
		self.dpdu = dpdu.copy()
		self.dpdv = dpdv.copy()
		self.dndu = dndu.copy()
		self.dndv = dndv.copy()

		self.nn = Normal._from_arr( dpdu._cross(dpdv).normalize() )
		self.u = uu
		self.v = vv
		self.shape = shape

		# adjust for handedness
		if shape is not None and \
			(shape.ro ^ shape.transform_swaps_handedness):
			self.nn *= -1.

		# for anti-aliasing
		self.dudx = self.dvdx = self.dudy = self.dvdy = 0.
		self.dpdx = self.dpdy = Vector(0., 0., 0.)

cdef void _diffgeom_compute_differential(DifferentialGeometry *self, RayDifferential ray)