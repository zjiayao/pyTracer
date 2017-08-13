"""
intersection.py

Implementation of Intersection.

v0.0
Created by Jiayao on July 30, 2017
Modifed on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo
import pytracer.transform as trans
import pytracer.reflection as refl
import pytracer.volume as vol
from pytracer.aggregate.primitive import Primitive

__all__ = ['Intersection']


class Intersection(object):
	def __init__(self, dg: 'geo.DifferentialGeometry'=None, pr: 'Primitive'=None,
			w2o: 'trans.Transform'=None, o2w: 'trans.Transform'=None, sh_id: INT=0, pr_id: INT=0, rEps: FLOAT=0.):
		self.dg = dg
		self.primitive = pr
		self.w2o = w2o
		self.o2w = o2w
		self.shapeId = sh_id
		self.primitiveId = pr_id
		self.rEps = rEps

	def __repr__(self):
		return "{}\nPrimitive ID: {}\nShape ID: {}\n".format(self.__class__, self.primitiveId, self.shapeId)

	def get_bsdf(self, ray: 'geo.RayDifferential') -> 'refl.BSDF':
		self.dg.compute_differential(ray)
		return self.primitive.get_BSDF(self.dg, self.o2w)

	def get_bssrdf(self, ray: 'geo.RayDifferential') -> 'vol.BSSRDF':
		self.dg.compute_differential(ray)
		return self.primitive.get_BSSRDF(self.dg, self.o2w)

	def le(self, w: 'geo.Vector') -> 'Spectrum':
		"""
		le()

		Compute the emitted radiance
		at a surface point intersected by
		a ray.
		"""
		area = self.primitive.get_area_light()
		if area is None:
			return Spectrum(0.)
		return area.l(self.dg.p, self.dg.nn, w)