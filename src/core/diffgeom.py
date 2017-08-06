'''
diffgeo.py

This module is part of the pyTracer, which
defines differential geometric operations.

v0.0
Created by Jiayao on July 28, 2017
'''
'''
imp.reload(src.core.pytracer)
imp.reload(src.core.geometry)
imp.reload(src.core.transform)
from src.core.pytracer import *
from src.core.geometry import *
from src.core.transform import *
'''

import numpy as np
from src.core.pytracer import *
from src.core.geometry import *
from src.core.shape import *

class DifferentialGeometry:
	'''
	Differential Geometry class
	'''
	def __init__(self, P: 'Point', DPDU: 'Vector', DPDV: 'Vector',
				 DNDU: 'Normal', DNDV: 'Normal', uu: 'FLOAT',
				 vv: 'FLOAT', sh: 'Shape'):
		self.p = P.copy()
		self.dpdu = DPDU.copy()
		self.dpdv = DPDV.copy()
		self.dndu = DNDU.copy()
		self.dndv = DNDV.copy()

		self.nn = Normal.fromVector(normalize(DPDU.cross(DPDV)))
		self.u = uu
		self.v = vv
		self.shape = sh

		# adjust for handedness
		if sh is not None and \
			(sh.ro ^ sh.transform_swaps_handedness):
			self.nn = -1. * self.nn

		# for anti-aliasing
		self.dudx = self.dvdx = self.dudy = self.dvdy = FLOAT(0.)
		self.dpdx = self.dpdy = Vector(0., 0., 0.)

	def __repr__(self):
		return "{}\nNormal Vector: {}".format(self.__class__, self.nn)

	def copy(self):
		return DifferentialGeometry(self.p, self.dpdu, self.dpdv,
				self.dndu, self.dndv, self.u, self.v, self.shape)

	@jit
	def compute_differential(self, ray:'RayDifferential'):
		if ray.has_differential:
			# estimate screen space change in p and (u, v)
			## compute intersections of incremental rays
			d = -self.nn.dot(self.p)

			#  dot product between Normal and Point is defined =D
			tx = -(self.nn.dot(ray.rxOrigin) + d) / (self.nn.dot(ray.rxDirection))
			px = ray.rxOrigin + tx * ray.rxDirection

			ty = -(self.nn.dot(ray.ryOrigin) + d) / (self.nn.dot(ray.ryDirection))
			py = ray.ryOrigin + ty * ray.ryDirection

			self.dpdx = px - self.p
			self.dpdy = py - self.p
			## compute (u, v) offsets
			### Init coefficients
			axes = [0, 1]
			if np.fabs(self.nn.x) > np.fabs(self.nn.y) and \
					np.fabs(self.nn.x) > np.fabs(self.nn.z):
				axes = [1, 2]
			elif np.fabs(self.nny) > np.fabs(self.nn.z):
				axes = [0, 2]

			A = [[self.dpdu[axes[0]], self.dpdv[axes[0]]],
				 [self.dpdu[axes[1]], self.dpdv[axes[1]]]]
			Bx = [px[axes[0]] - self.p[axes[0]],
				  px[axes[1]] - self.p[axes[1]]]
			By = [py[axes[0]] - self.p[axes[0]],
				  py[axes[1]] - self.p[axes[1]]]

			try:
				[self.dudx, self.dvdx] = np.linalg.solve(A, Bx)
			except numpy.linalg.linalg.LinAlgError:
				[self.dudx, self.dvdx] = [0., 0.]

			try:
				[self.dudy, self.dvdy] = np.linalg.solve(A, By)
			except numpy.linalg.linalg.LinAlgError:
				[self.dudy, self.dvdy] = [0., 0.]			

		else:
			self.dudx = self.dvdx = self.dudy = self.dvdy = FLOAT(0.)
			self.dpdx = self.dpdy = Vector(0., 0., 0.)














