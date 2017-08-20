"""
primitive.py

Implementation of primitives.


v0.0
Created by Jiayao on July 30, 2017
Modified on Aug 13, 2017
"""

from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
import pytracer.geometry as geo
import pytracer.transform as trans
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from pytracer.aggregate import Intersection

__all__ = ['Primitive', 'GeometricPrimitive', 'TransformedPrimitive']


class Primitive(object, metaclass=ABCMeta):

	#__primitiveId = 0
	next_primitiveId = 1

	def __init__(self):
		self.primitiveId = Primitive.next_primitiveId
		Primitive.next_primitiveId += 1

	def __repr__(self):
		return "{}\nPrimitive ID: {}\nNext Primitive ID: {}" \
			.format(self.__class__, self.primitiveId, Primitive.next_primitiveId)

	@abstractmethod
	def world_bound(self) -> 'geo.BBox':
		raise NotImplementedError('{}.world_bound(): Not implemented'.format(self.__class__))

	@abstractmethod
	def can_intersect(self) -> bool:
		raise NotImplementedError('{}.can_intersect(): Not implemented'.format(self.__class__))

	@abstractmethod
	def intersect(self, r: 'geo.Ray', isect: 'Intersection') -> bool:
		raise NotImplementedError('{}.intersect(): Not implemented'.format(self.__class__))

	@abstractmethod
	def intersect_p(self, r : 'geo.Ray') -> bool:
		raise NotImplementedError('{}.intersect_p(): Not implemented'.format(self.__class__))

	@abstractmethod
	def refine(self, refined: ['Primitive']):
		raise NotImplementedError('{}.refine(): Not implemented'.format(self.__class__))

	@abstractmethod
	def get_area_light(self):
		raise NotImplementedError('{}.get_area_light(): Not implemented'.format(self.__class__))

	@abstractmethod
	def get_bsdf(self, dg: 'geo.DifferentialGeometry', o2w: 'trans.Transform') -> 'BSDF':
		raise NotImplementedError('{}.get_bsdf(): Not implemented'.format(self.__class__))

	@abstractmethod
	def get_bssrdf(self, dg: 'geo.DifferentialGeometry', o2w: 'trans.Transform') -> 'BSSRDF':
		raise NotImplementedError('{}.get_bssrdf(): Not implemented'.format(self.__class__))

	def full_refine(self, refined: ['Primitive']):
		todo = [self]
		while len(todo) > 0:
			prim = todo[-1]
			del todo[-1]
			if prim.can_intersect():
				refined.append(prim)
			else:
				prim.refine(todo)
	
	
# Shapes to be rendered directly
class GeometricPrimitive(Primitive):
	def __init__(self, s: 'sh.Shape', m: 'Material', a: 'AreaLight' = None):
		super().__init__()
		self.shape = s
		self.material = m
		self.areaLight = a

	def __repr__(self):
		return super().__repr__() + '\n{}'.format(self.shape)

	def intersect(self, r: 'geo.Ray', isect: 'Intersection') -> bool:
		# it is the caller's responsibility to ensure
		# isect is not None
		from pytracer.aggregate import Intersection
		is_intersect, thit, rEps, dg = self.shape.intersect(r)
		if not is_intersect:
			return False

		isect.dg = dg
		isect.primitive = self
		isect.w2o = self.shape.w2o
		isect.o2w = self.shape.o2w
		isect.shapeId = self.shape.shapeId
		isect.primitiveId = self.primitiveId
		isect.rEps = rEps

		# isect = Intersection(dg, self, self.shape.w2o, self.shape.o2w,
			# self.shape.shapeId, self.primitiveId, rEps)
		r.maxt = thit
		return True

	def intersect_p(self, r: 'geo.Ray') -> bool:
		return self.shape.intersect_p(r)

	def world_bound(self) -> 'geo.BBox':
		return self.shape.world_bound()

	def can_intersect(self) -> bool:
		return self.shape.can_intersect()

	def refine(self, refined: ['Primitive']):
		r = self.shape.refine()
		for sh in r:
			refined.append(GeometricPrimitive(sh, self.material, self.areaLight))

	def get_area_light(self):
		return self.areaLight

	def get_bsdf(self, dg: 'geo.DifferentialGeometry', o2w: 'trans.Transform'):
		dgs = self.shape.get_shading_geometry(o2w, dg)
		return self.material.get_bsdf(dg, dgs)

	def get_bssrdf(self, dg: 'geo.DifferentialGeometry', o2w: 'trans.Transform'):
		dgs = self.shape.get_shading_geometry(o2w, dg)
		return self.material.get_bssrdf(dg, dgs)


# sh.Shapes with animated transfomration and object instancing
class TransformedPrimitive(Primitive):
	def __init__(self, prim: 'Primitive', w2p: 'Animatedtrans.Transform'):
		super().__init__()
		self.primitive = prim
		self.w2p = w2p

	def __repr__(self):
		return super().__repr__(self) + '\n{}'.format(self.prim)

	def intersect(self, r: 'geo.Ray', isect: 'Intersection') -> bool:
		w2p = self.w2p.interpolate(r.time)
		ray = w2p(r)
		is_intersect = self.primitive.intersect(ray, isect)
		if not is_intersect:
			return False

		r.maxt = ray.maxt

		isect.primitiveId = self.primitiveId

		if not w2p.is_identity():
			isect.w2o = isect.w2o * w2p
			isect.o2w = isect.w2o.inverse()

			p2w = w2p.inverse()
			dg = isect.dg
			dg.p = p2w(dg.p)
			dg.nn = geo.normalize(p2w(dg.nn))
			dg.dpdu = p2w(dg.dpdu)
			dg.dpdv = p2w(dg.dpdv)
			dg.dndu = p2w(dg.dndu)
			dg.dndv = p2w(dg.dndv)

		return True

	def intersect_p(self, r: 'geo.Ray') -> bool:
		return self.primitive.intersect_p(self.w2p(r))		

	def world_bound(self) -> 'geo.BBox':
		return self.w2p.motion_bounds(self.primitive.world_bound(), True)

	def can_intersect(self) -> bool:
		return self.shape.can_intersect()

	def refine(self, refined: ['Primitive']):
		r = self.shape.refine()
		for sh in r:
			refined.append(GeometricPrimitive(sh, self.material, self.areaLight))


