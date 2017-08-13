"""
path.py

under pytracer.integrator.surface package

Models path transport

Created by Jiayao on Aug 9, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo
import pytracer.aggregate as agg
import pytracer.sampler as spler
import pytracer.scene as scn
import pytracer.renderer as ren
import pytracer.reflection as refl
import pytracer.light as lgt
from pytracer.integrator.surface import SurfaceIntegrator

__all__ = ['PathIntegrator']


class PathIntegrator(SurfaceIntegrator):
	"""
	PathIntegrator

	Path tracing using Russian roulette.
	Also support maximum depth.
	"""
	SAMPLE_DEPTH = 3

	def __init__(self, max_depth: INT=5):
		self.max_depth = max_depth
		self.__light_num_offset = [0 for _ in range(PathIntegrator.SAMPLE_DEPTH)]
		self.__light_sample_offsets = [None for _ in range(PathIntegrator.SAMPLE_DEPTH)]
		self.__bsdf_sample_offsets = [None for _ in range(PathIntegrator.SAMPLE_DEPTH)]
		self.__path_sample_offsets = [None for _ in range(PathIntegrator.SAMPLE_DEPTH)]

	def request_samples(self, sampler: 'spler.Sample', sample: 'spler.Sample', scene: 'scn.Scene'):
		# after first few bounces switches to uniform random
		for i in range(self.SAMPLE_DEPTH):
			self.__light_sample_offsets[i] = lgt.LightSampleOffset(1, sample)
			self.__light_num_offset[i] = sample.add_1d(1)
			self.__bsdf_sample_offsets[i] = refl.BSDFSampleOffset(1, sample)
			self.__path_sample_offsets[i] = refl.BSDFSampleOffset(1, sample)

	def li(self, scene: 'scn.Scene', renderer: 'ren.Renderer', r: 'geo.RayDifferential',
			isect: 'agg.Intersection', sample: 'spler.Sample', rng=np.random.rand) -> 'Spectrum':
		# common variables
		# product of refl.BSDF and cosines for vertices
		# generated so far, divided by pdf's
		path_throughput = Spectrum(1.)

		# radiance from the running total of amount of
		# scattered ($\sum P(\bar{p_i})$)
		L = Spectrum(0.)

		# next ray to be traced to extend the path
		ray = r#geo.RayDifferential.fromRD(r)

		# records if last outgoing direction sample
		# was due to specular reflection
		specular_bounce = False

		# most recently added vertex
		isectp = isect

		# subsequent vertex
		local_isect = agg.Intersection()
		print('+')
		bounce_cnt = 0
		while True:
			# add possibly emitted light
			if bounce_cnt == 0 or specular_bounce:
				print('++')
				# emitted light is included by
				# previous tracing via direct lighting
				# exceptions for the first tracing or sampling a
				# specular direction since it is omitted from estimate_direct()
				L += path_throughput * isectp.le(-ray.d)

			# sample illumination from lights
			bsdf = isectp.get_BSDF(ray)
			p = bsdf.dgs.p
			n = bsdf.dgs.nn
			wo = -ray.d

			if bounce_cnt < PathIntegrator.SAMPLE_DEPTH:
				print('bounce_cnt: ', bounce_cnt)
				# use samples
				from pytracer.integrator.utility import uniform_sample_one_light
				L += path_throughput * uniform_sample_one_light(scene, renderer, p, n, wo,
						isectp.rEps, ray.time, bsdf, sample, self.__light_num_offset[bounce_cnt],
						self.__light_sample_offsets[bounce_cnt], self.__bsdf_sample_offsets[bounce_cnt], rng=rng)
			else:
				# use uniform random
				from pytracer.integrator.utility import uniform_sample_one_light
				L += path_throughput * uniform_sample_one_light(scene, renderer, p, n, wo,
						isectp.rEps, ray.time, bsdf, sample, rng=rng)
			print('++++')
			# sample refl.BSDF to get new direction
			# get refl.BSDFSample for new direction
			if bounce_cnt < PathIntegrator.SAMPLE_DEPTH:
				out_bsdf_smp = refl.BSDFSample.from_sample(sample, self.__path_sample_offsets[bounce_cnt], 0)
			else:
				out_bsdf_smp = refl.BSDFSample.from_rand(rng)

			pdf, wi, flags, f = bsdf.sample_f(wo, out_bsdf_smp, refl.BDFType.ALL)

			if f.is_black() or pdf == 0.:
				break
			print('+++++')
			specular_bounce = (flags & refl.BDFType.SPECULAR) != 0
			path_throughput *= f * wi.abs_dot(n) / pdf
			ray = geo.RayDifferential(p, wi, ray, isectp.rEps)
			print('#')
			# possibly terminate
			if bounce_cnt > PathIntegrator.SAMPLE_DEPTH:
				cont_prob = min(.5, path_throughput.y())	# high prob for terminating for low contribution paths
				if rng() > cont_prob:
					break

				# otherwise apply Russian roulette
				path_throughput /= cont_prob

			if bounce_cnt == self.max_depth:
				break
			print('##')
			# find next vertex
			hit, local_isect = scene.intersect(ray)
			if not hit:
				# ambient light
				if specular_bounce:
					for light in scene.lights:
						L += path_throughput * light.le(ray)
				break
			print('###')
			if bounce_cnt > 1:
				path_throughput *= renderer.transmittance(scene, ray, None, rng)

			isectp = local_isect

			bounce_cnt += 1

		return L