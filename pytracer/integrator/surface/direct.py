"""
direct.py

under pytracer.integrator.surface package

Models direct lighting integrator

Created by Jiayao on Aug 9, 2017
"""
from __future__ import absolute_import
from enum import Enum
from pytracer import *
import pytracer.geometry as geo
from pytracer.integrator.surface import SurfaceIntegrator

__all__ = ['LightStrategy', 'DirectLightingIntegrator']


class LightStrategy(Enum):
	"""
	LightStrategy Class

	Enum for light strategy
	for direct lighting
	"""
	SAMPLE_ALL_UNIFORM = 1
	SAMPLE_ONE_UNIFORM = 2


class DirectLightingIntegrator(SurfaceIntegrator):
	"""
	DirectLightingIntegrator

	`SurfaceIntegrator` using direct lighting
	"""
	def __init__(self, strategy: 'LightStrategy'=LightStrategy.SAMPLE_ALL_UNIFORM, max_depth: INT=5):
		self.strategy = strategy
		self.max_depth = max_depth
		self.light_num_offset = 0
		self.light_sample_offsets = None
		self.bsdf_sample_offsets = None

	def __repr__(self):
		return "{}\nStrategy: {}\n".format(self.__class__, self.strategy)

	def request_samples(self, sampler: 'Sampler', sample: 'Sample', scene: 'Scene'):
		from pytracer.light import LightSampleOffset
		from pytracer.reflection import BSDFSampleOffset
		if self.strategy == LightStrategy.SAMPLE_ALL_UNIFORM:
			# sampling all lights
			n_lights = len(scene.lights)
			self.light_sample_offsets = []
			self.bsdf_sample_offsets = []
			for _, light in enumerate(scene.lights):
				n_smp = light.ns
				if sampler is not None:
					n_smp = sampler.round_size(n_smp)

				self.light_sample_offsets.append(LightSampleOffset(n_smp, sample))
				self.bsdf_sample_offsets.append(BSDFSampleOffset(n_smp, sample))

			self.light_num_offset = -1
		else:
			# sampling one light
			n_lights = len(scene.lights)
			self.light_sample_offsets = [LightSampleOffset(1, sample)]
			self.light_num_offset = sample.add1d(1)
			self.bsdf_sample_offsets = [BSDFSampleOffset(1, sample)]


	def li(self, scene: 'Scene', renderer: 'Renderer', ray: 'geo.RayDifferential',
			isect: 'Intersection', sample: 'Sample', rng=np.random.rand) -> 'Spectrum':
		L = Spectrum(0.)
		# Evaluate BSDF at hit point
		bsdf = isect.get_bsdf(ray)

		# Init
		p = bsdf.dgs.p
		n = bsdf.dgs.nn
		wo = -ray.d

		# compute emitted light if ray hit light src
		L += isect.le(wo)

		# compute direct lighting
		if len(scene.lights) > 0:
			if self.strategy == LightStrategy.SAMPLE_ALL_UNIFORM:
				from pytracer.integrator.utility import uniform_sample_all_lights
				L += uniform_sample_all_lights(scene, renderer, p, n, wo, isect.rEps, ray.time,
						bsdf, sample, self.light_sample_offsets, self.bsdf_sample_offsets, rng)

			elif self.strategy == LightStrategy.SAMPLE_ONE_UNIFORM:
				from pytracer.integrator.utility import uniform_sample_one_light
				L += uniform_sample_one_light(scene, renderer, p, n, wo, isect.rEps, ray.time,
						bsdf, sample, self.light_num_offset, self.light_sample_offsets, self.bsdf_sample_offsets, rng)

			else:
				raise RuntimeError("Unknown LightStrategy")

		if ray.depth + 1 < self.max_depth:
			# trace rays for reflection and refraction
			from pytracer.integrator.utility import (specular_reflect, specular_transmit)
			L += specular_reflect(ray, bsdf, isect, renderer, scene, sample, rng)
			L += specular_transmit(ray, bsdf, isect, renderer, scene, sample, rng)

		return L

