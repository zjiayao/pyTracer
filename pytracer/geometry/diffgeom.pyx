"""
diffgeo.pyx

This module is part of the pyTracer, which
implemented differential geometric operations.

v0.0
Created by Jiayao on July 28, 2017
Modified on Aug 13, 2017
Cythonized on Aug 30, 2017
"""

__all__ = ['DifferentialGeometry']


cdef class DifferentialGeometry:
	def __cinit__(self, Point p=Point(), Vector dpdu=Vector(), Vector dpdv=Vector(), Normal dndu=Normal(),
	              Normal dndv=Normal(), FLOAT_t uu=0., FLOAT_t vv=0., Shape shape=None):
		self.p = p
		self.dpdu = dpdu
		self.dpdv = dpdv
		self.dndu = dndu
		self.dndv = dndv

		self.nn = Normal._from_arr( dpdu._cross(dpdv)._normalize() )
		self.u = uu
		self.v = vv
		self.shape = shape

		# adjust for handedness
		if shape is not None and \
			(shape.ro ^ shape.transform_swaps_handedness):
			self.nn *= -1.

		# for anti-aliasing
		self.dudx = self.dvdx = self.dudy = self.dvdy = 0.
		self.dpdx = Vector(0., 0., 0.)
		self.dpdy = Vector(0., 0., 0.)

	cdef void _compute_differential(self, RayDifferential ray):
		cdef:
			Normal d
			FLOAT_t inv_tx = self.nn.dot(ray.rxDirection)
			FLOAT_t inv_ty = self.nn.dot(ray.ryDirection)
			FLOAT_t tx, ty
			Point px, py
			INT_t axes[2]
			FLOAT_t A[2][2]
			FLOAT_t Bx[2]
			FLOAT_t By[2]

		if not ray.has_differentials or is_zero(inv_tx) or is_zero(inv_ty):
			self.dudx = self.dvdx = self.dudy = self.dvdy = 0.
			self.dpdx = Vector(0., 0., 0.)
			self.dpdy = Vector(0., 0., 0.)

		else:
			d = -self.nn.dot(self.p)
			tx = -(self.nn.dot(ray.rxOrigin) + d) / inv_tx
			ty = -(self.nn.dot(ray.ryOrigin) + d) / inv_ty
			px = ray.rxOrigin + ray.rxDirection * tx
			py = ray.ryOrigin + ray.ryDirection * ty

			self.dpdx = px - self.p
			self.dpdx = py - self.p
			assert isinstance(self.dpdx, Vector)
			axes[0] = 0
			axes[1] = 1

			if fabs(self.nn.x) > fabs(self.nn.y) and fabs(self.nn.x) > fabs(self.nn.z):
				axes[0] = 1
				axes[1] = 2
			elif fabs(self.nn.y) > fabs(self.nn.z):
				axes[1] = 2

			A[0][0] = self.dpdu[axes[0]]
			A[0][1] = self.dpdv[axes[0]]
			A[1][0] = self.dpdu[axes[1]]
			A[1][1] = self.dpdv[axes[1]]

			Bx[0] = px[axes[0]] - self.p[axes[0]]
			Bx[1] = px[axes[1]] - self.p[axes[1]]

			By[0] = py[axes[0]] - self.p[axes[0]]
			By[1] = py[axes[1]] - self.p[axes[1]]

			if not solve_linear_2x2(A, Bx, &self.dudx, &self.dvdx):
				self.dudx = 0.
				self.dvdx = 0.

			if not solve_linear_2x2(A, By, &self.dudy, &self.dvdy):
				self.dudy = 0.
				self.dvdy = 0.

	cpdef void compute_differential(self, RayDifferential ray):
		self._compute_differential(ray)
