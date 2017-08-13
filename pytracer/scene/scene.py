"""
scene.py

Scene Class

Created by Jiayao on Aug 5, 2017
"""
from src.accelerator.primitive import *  # Primitive, Intersection
from src.light.light import *  # Light
from src.volume.volume import *  # VolumeRegion


class Scene(object):
	"""
	Scene Class
	"""
	def __init__(self, agg: 'Primitive', lights: ['Light'],
						vr: ['VolumeRegion']):
		self.agg = agg
		self.lights = lights
		self.vr = vr

		self.bound = self.agg.world_bound()
		if self.vr is not None:
			self.bound.union(vr.world_bound())

	def __repr__(self):
		return "{}\nAggregates: {}\nLights: {}\nVolume Regions: {}" \
					.format(self.__class__, self.agg, len(self.lights), "N/A" if self.vr is None else len(self.vr))

	def intersect(self, ray: 'Ray') -> [bool, 'Intersection']:
		return self.agg.intersect(ray)

	def intersectP(self, ray: 'Ray') -> bool:
		return self.agg.intersect_p(ray)

	def world_bound(self) -> 'BBox':
		return self.bound