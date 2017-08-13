"""
whitted.py

under pytracer.integrator.surface package

Models Whitted integrator

Created by Jiayao on Aug 9, 2017
"""
from __future__ import absolute_import
from enum import Enum
from pytracer import *
import pytracer.geometry as geo
import pytracer.aggregate as agg
import pytracer.sampler as spler
import pytracer.scene as scn
import pytracer.renderer as ren
from pytracer.integrator.surface import SurfaceIntegrator


class WhittedIntegrator(SurfaceIntegrator):
	"""
	WhittedIntegrator Class
	"""
	def __init__(self, max_depth:INT=5):
		self.max_depth = max_depth

	def __repr__(self):
		return "{}\nMax Depth: {}\n".format(self.__class__, self.max_depth)

	def li(self, scene: 'scn.Scene', renderer: 'ren.Renderer', ray: 'geo.RayDifferential',
			isect: 'agg.Intersection', sample: 'spler.Sample', rng=np.random.rand) -> 'Spectrum':
		L = Spectrum(0.)
		# Evaluate refl.BSDF at hit point
		bsdf = isect.get_BSDF(ray)

		# Init
		p = bsdf.dgs.p
		n = bsdf.dgs.nn
		wo = -ray.d

		# compute emitted light if ray hit light src
		L += isect.le(wo)

		# iterate through all lights
		for lgt in scene.lights:
			Li, wi, pdf, vis = lgt.sample_l(p, isect.rEps, lgt.LightSample.from_rand(rng), ray.time)
			if Li.is_black() or pdf == 0.:
				continue

			# add contribution
			f = bsdf.f(wo, wi)
			if not f.is_black() and vis.unoccluded(scene):
				L += f * Li * wi.abs_dot(n) * vis.transmittance(scene, renderer, sample, rng) / pdf

		if ray.depth + 1 < self.max_depth:
			# trace rays for reflection and refraction
			from pytracer.integrator.utility import (specular_reflect, specular_transmit)
			L += specular_reflect(ray, bsdf, isect, renderer, scene, sample, rng)
			L += specular_transmit(ray, bsdf, isect, renderer, scene, sample, rng)

		return L
