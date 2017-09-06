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
# cyclic declaration
cdef class DifferentialGeometry
from pytracer.shape.shape cimport Shape


cdef class DifferentialGeometry:
	cdef:
		Point p
		Vector dpdu, dpdv, dpdx, dpdy
		Normal dndu, dndv, nn
		FLOAT_t u, v, dudx, dudy, dvdx, dvdy
		# Shape shape

	cdef void _compute_differential(self, RayDifferential ray)
	cpdef void compute_differential(self, RayDifferential ray)