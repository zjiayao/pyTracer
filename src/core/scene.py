'''
scene.py

Scene Class

Created by Jiayao on Aug 5, 2017
'''
from numba import jit
import numpy as np
from src.core.pytracer import *
from src.core.primitive import *
from src.core.light import *


class Scene(object):
	'''
	Scene Class
	'''
	def __init__(self, agg: 'Primitive', lights: ['Light'],
						vr: ['VolumeRegion'] ):
		self.agg = agg
		self.lights = lights
		self.vr = vr

		self.bound = self.agg.world_bound()
		if self.vr is not None:
			bound.union(vr.world_bound())

	def __repr__(self):
		return "{}\nAggregates: {}\nLights: {}\n Volume Regions: {}" \
					.format(self.__class__, self.agg, len(self.lights), len(self.vr))

	def intersect(self, ray: 'Ray') -> [bool, 'Intersection']:
		return self.agg.intersect(ray)

	def intersectP(self, ray: 'Ray') -> bool:
		return self.agg.intersectP(ray)

	def world_bound(self) -> 'BBox':
		return self.bound