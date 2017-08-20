"""
scene.py

Scene Class

Created by Jiayao on Aug 5, 2017
"""
from __future__ import absolute_import
import pytracer.geometry as geo
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from pytracer.aggregate import Intersection

__all__ = ['Scene']


class Scene(object):
	"""
	Scene Class
	"""
	def __init__(self, aggregate: 'Primitive', lights: ['Light'],
	             vr: ['VolumeRegion']):
		self.aggregate = aggregate
		self.lights = lights
		self.vr = vr

		self.bound = self.aggregate.world_bound()
		if self.vr is not None:
			self.bound.union(vr.world_bound())

	def __repr__(self):
		return "{}\nAggregates: {}\nLights: {}\nVolume Regions: {}" \
					.format(self.__class__, self.aggregate, len(self.lights), "N/A" if self.vr is None else len(self.vr))

	def intersect(self, ray: 'geo.Ray', isect: 'Intersection') -> bool:
		return self.aggregate.intersect(ray, isect)

	def intersect_p(self, ray: 'geo.Ray') -> bool:
		return self.aggregate.intersect_p(ray)

	def world_bound(self) -> 'geo.BBox':
		return self.bound