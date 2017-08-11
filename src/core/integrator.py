'''
integrator.py

Model integrators.

Created by Jiayao on Aug 9, 2017
'''

from numba import jit
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
from src.core.pytracer import *
from src.core.geometry import *
from src.core.scene import *
from src.core.camera import *
from src.core.renderer import *
from src.core.sampler import *
from src.core.spectrum import *
from src.core.light import *
from src.core.reflection import *
from src.core.montecarlo import *

# Utility Functions

@jit
def compute_light_sampling_cdf(scene: 'Scene') -> 'Distribution1D':
	'''
	compute_light_sampling_cdf()

	Creates a one dimensional
	distribution based on the power
	of all lights in the scene.
	'''
	n_lights = len(scene.lights)
	power = []
	for light in scene.lights:
		power.append(light.power(scene).y())

	return Distribution1D(power)

@jit
def uniform_sample_all_lights(scene: 'Scene', renderer: 'Renderer', p: 'Point', n: 'Normal', 
		wo: 'Vector', rEps: FLOAT, time: FLOAT, bsdf: 'BSDF', sample: 'Sample', light_offsets: ['LightSampleOffset'], 
		bsdf_offsets: ['BSDFSampleOffset'], rng=np.random.rand):
	'''
	uniform_sample_all_lights()

	Estimate contribution by each light
	individually using MC.
	'''
	L = Spectrum(0.)

	for i, light in enumerate(scene.lights):
		n_smp = 1 if light_sample_offsets is None else light_sample_offsets[i].nSamples
		# estimate direct lighting
		## find light and bsdf for estimating
		Ld = Spectrum(0.)
		for j in range(n_smp):
			if light_sample_offsets is not None and bsdf_sample_offsets is not None:
				light_smp = LightSample.fromSample(sample, light_sample_offsets[i], j)
				bsdf_smp = BSDFSample.fromSample(sample, bsdf_sample_offsets[i], j)
			else:
				light_smp = LightSample.fromRand(rng)
				bsdf_smp = BSDFSample.fromRand(rng)

			Ld += estimate_direct(scene, renderer, light, p, n, wo, rEps, time, bsdf,
						light_smp, bsdf_smp, BDFType(BDFType.ALL & ~BDFType.SPECULAR), rng)

		L += Ld / n_smp

	return L


@jit
def uniform_sample_one_light(scene: 'Scene', renderer: 'Renderer', p: 'Point', n: 'Normal', 
		wo: 'Vector', rEps: FLOAT, time: FLOAT, bsdf: 'BSDF', sample: 'Sample', light_num_offset: INT=-1,
		light_offsets: ['LightSampleOffset']=None, bsdf_offsets: ['BSDFSampleOffset']=None, rng=np.random.rand):
	'''
	uniform_sample_one_light()

	random choose one light to sample
	and multiply with number of lights
	to compensate on average
	'''
	# choose a light
	# light power based importance sampling
	# is to implement
	n_lights = len(scene.lights)
	if n_lights == 0:
		return Spectrum(0.)

	if light_num_offset == -1:
		# use random
		light_num = ftoi(rand() * n_lights)
	else:
		light_num = ftoi(sample.oneD[light_num_offset][0] * n_lights)

	light_num = min(light_num, n_lights - 1)
	light = scene.lights[light_num]

	# init
	if light_sample_offsets is not None and bsdf_sample_offsets is not None:
		light_smp = LightSample.fromSample(sample, light_sample_offsets[0], 0)
		bsdf_smp = BSDFSample.fromSample(sample, bsdf_sample_offsets[0], 0)
	else:
		light_smp = LightSample.fromRand(rng)
		bsdf_smp = BSDFSample.fromRand(rng)	


	
	return n_lights * estimate_direct(scene, renderer, light, p, n, wo, rEps, time, bsdf,
						light_smp, bsdf_smp, BDFType(BDFType.ALL & ~BDFType.SPECULAR), rng)


@jit
def specular_reflect(ray: 'RayDifferential', bsdf: 'BSDF', isect: 'Intersection',
			renderer: 'Renderer', scene: 'Scene', sample: 'Sample', rng) -> 'Spectrum':
	wo = -ray.d
	p = bsdf.dgs.p
	n = bsdf.dgs.nn
	pdf, wi, _, f = bsdf.sample_f(wo, BSDFSample.fromRand(rng), BDFType(BDFType.REFLECTION | BDFType.SPECULAR))
	L = Spectrum(0.)
	if pdf > 0. and not f.is_black() and wi.abs_dot(n) != 0.:
		# compute ray differential
		rd = RayDifferential(p, wi, ray, isect.rEps)
		if ray.hasDifferentials:
			rd.hasDifferentials = True
			rd.rxOrigin = p + isect.dg.dpdx
			rd.ryOrigin = p + isect.dg.dpdy

			# compute reflected directions of differentials
			dndx = bsdf.dgs.dndu * bsdf.dgs.dudx + \
				   bsdf.dgs.dndv * bsdf.dgs.dvdx
			dndy = bsdf.dgs.dndu * bsdf.dgs.dudy + \
				   bsdf.dgs.dndv * bsdf.dgs.dvdy				   

			dwodx = -ray.rxDirection - wo
			dwody = -ray.ryDirection - wo
			dDNdx = dwodx.dot(n) + wo.dot(dndx)
			dDNdy = dwody.dot(n) + wo.dot(dndy)

			rd.rxDirection = wi - dwodx + 2. * Vector.fromNormal(wo.dot(n) * dndx + dDNdx * n)
			rd.ryDirection = wi - dwody + 2. * Vector.fromNormal(wo.dot(n) * dndy + dDNdy * n)

		Li = renderer.li(scene, rd, sample, rng)
		L = f * Li * wi.abs_dot(n) / pdf

	return L

@jit
def specular_transmit(ray: 'RayDifferential', bsdf: 'BSDF', isect: 'Intersection',
			renderer: 'Renderer', scene: 'Scene', sample: 'Sample', rng) -> 'Spectrum':
	wo = -ray.d
	p = bsdf.dgs.p
	n = bsdf.dgs.nn
	pdf, wi, _, f = bsdf.sample_f(wo, BSDFSample.fromRand(rng), BDFType(BDFType.TRANSMISSION | BDFType.SPECULAR))
	L = Spectrum(0.)
	if pdf > 0. and not f.is_black() and wi.abs_dot(n) != 0.:
		# compute ray differential
		rd = RayDifferential(p, wi, ray, isect.rEps)
		if ray.hasDifferentials:
			rd.hasDifferentials = True
			rd.rxOrigin = p + isect.dg.dpdx
			rd.ryOrigin = p + isect.dg.dpdy

			eta = bsdf.eta
			w = -wo
			if wo.dot(n) < 0.:
				eta = 1. / eta

			# compute reflected directions of differentials
			dndx = bsdf.dgs.dndu * bsdf.dgs.dudx + \
				   bsdf.dgs.dndv * bsdf.dgs.dvdx
			dndy = bsdf.dgs.dndu * bsdf.dgs.dudy + \
				   bsdf.dgs.dndv * bsdf.dgs.dvdy				   

			dwodx = -ray.rxDirection - wo
			dwody = -ray.ryDirection - wo
			dDNdx = dwodx.dot(n) + wo.dot(dndx)
			dDNdy = dwody.dot(n) + wo.dot(dndy)

			mu = eta * w.dot(n) - wi.dot(n)
			dmudx = (eta - (eta * eta * w.dot(n) / wi.dot(n))) * dDNdx
			dmudy = (eta - (eta * eta * w.dot(n) / wi.dot(n))) * dDNdy

			rd.rxDirection = wi + eta * dwodx - Vector.fromNormal(mu * dndx + dmudx * n)
			rd.ryDirection = wi + eta * dwody - Vector.fromNormal(mu * dndy + dmudy * n)

		Li = renderer.li(scene, rd, sample, rng)
		L = f * Li * wi.abs_dot(n) / pdf

	return L


@jit
def estimate_direct(scene: 'Scene', renderer: 'Renderer', light: 'Light', p:'Point', n: 'Normal', 
			wo: 'Vector', rEps: FLOAT, time: FLOAT, bsdf: 'BSDF', light_smp: 'LightSample', bsdf_smp: 'BSDFSample',
			flags: 'BDFType', rng=np.random.rand) -> 'Spectrum':
	'''
	estimate_direct()

	Estimate direct lighting
	using MC Multiple Importance Sampling.
	'''
	Ld = Spectrum(0.)
	# sample light source
	Li, wi, light_pdf, vis = light.sample_l(p, rEps, light_smp, time)
	if light_pdf > 0. and not Li.is_black():
		f = bsdf.f(wo, wi, flags)
		if not f.is_black() and vis.unoccluded(scene):
			# add contribution to reflected radiance
			## account for attenuation due to participating media,
			## left for `VolumeIntegrator` to do
			Li *= vis.transmittance(scene, renderer, None, rng) 
			if light.is_delta_light():
				Ld += f * Li * (wi.abs_dot(n) / light_pdf)
			else:
				bsdf_pdf = bsdf.pdf(wo, wi, flags)
				weight = power_heuristic(1, light_pdf, 1, bsdf_pdf) # Power heuristic for HG phase function
				Ld += f * Li * (wi.abs_dot(n) * weight / light_pdf)

	# sample BSDF
	## no need if delta light
	if not light.is_delta_light():
		bsdf_pdf, wi, smp_type, f = bsdf.sample_f(wo, bsdf_smp, flags)
		if not f.is_black() and bsdf_pdf > 0.:
			wt = 1.
			if not smp_type & BDFType.SPECULAR:	# MIS not apply to specular direction
				light_pdf = light.pdf(p, wi)
				if light_pdf == 0.:
					return Ld
				wt = power_heuristic(1, bsdf_pdf, 1, light_pdf)
				# add contribution
				Li = Spectrum(0.)
				ray = RayDifferential(p, wi, rEps, np.inf, time)
				hit, light_isect = scene.intersect(ray)
				if hit:
					if light_isect.primitive.get_area_light() == light:
						Li = light_isect.le(-wi)
				else:
					Li = light.le(ray) # light illum.

				if not Li.is_black():
					Li *= renderer.transmittance(scene, ray, None, rng) # attenuation
					Ld += f * Li * wi.abs_dot(n) * wt / bsdf_pdf

	return Ld



# Integrator Classes
class Integrator(object, metaclass=ABCMeta):
	'''
	Integrator Class

	Base Class for Integrators
	'''
	def __repr__(self):
		return "{}\n".format(self.__class__)

	# optional preprocessing
	def preprocess(self, scene: 'Scene', camera: 'Camera', renderer:'Renderer'):
		pass

	# optional requests for samples
	def request_samples(self, sampler: 'Sampler', sample: 'Sample', scene: 'Scene'):
		pass


class SurfaceIntegrator(Integrator):
	'''
	SurfaceIntegrator Class

	Interface for surface integrators
	'''

	@abstractmethod
	def li(self, scene: 'Scene', renderer: 'Renderer', ray: 'RayDifferential',
			isect: 'Intersection', sample: 'Sample', rng=np.random.rand) -> 'Spectrum':
		'''
		li()

		Returns the outgoing radiance at the intersection
		point of a given ray with scene.
		- scene
			`Scene` to be rendered
		- renderer
			`Renderer` used for rendering, `li()` or `transmittance()` might
			be called
		- ray
			`Ray` to evaluate incident radiance
		- isect
			First `Intersection` of the ray in the `Scene`
		- sample
			A `Sample` generated by a `Sampler` for this ray.
			Might be used for MC methods.
		- rng
			Random number generator, by default `numpy.random.rand`
		'''
		raise NotImplementedError('src.core.integrator.{}.li(): abstract method '
						'called'.format(self.__class__)) 	

class WhittedIntegrator(SurfaceIntegrator):
	'''
	WhittedIntegrator Class
	'''
	def __init__(self, max_depth:INT=5):
		self.max_depth = max_depth

	def __repr__(self):
		return "{}\nMax Depth: {}\n".format(self.__class__, self.max_depth)

	@jit
	def li(self, scene: 'Scene', renderer: 'Renderer', ray: 'RayDifferential',
			isect: 'Intersection', sample: 'Sample', rng=np.random.rand) -> 'Spectrum':
		L = Spectrum(0.)
		# Evaluate BSDF at hit point
		bsdf = isect.get_BSDF(ray)

		# Init
		p = bsdf.dgs.p
		n = bsdf.dgs.nn
		wo = -ray.d

		# compute emitted light if ray hit light src
		L += isec.le(wo)

		# iterate through all lights
		for lgt in scene.lights:
			Li, wi, pdf, vis = lgt.sample_l(p, isec.rEps, LightSample.fromRand(rng), ray.time)
			if Li.is_black() or pdf == 0.:
				continue

			# add contribution
			f = bsdf.f(wo, wi)
			if not f.is_black() and vis.unoccluded(scene):
				L += f * Li * wi.abs_dot(n) * vis.transmittance(scene, renderer, sample, rng) / pdf

		if ray.depth + 1 < self.max_depth:
			# trace rays for reflection and refraction
			L += specular_reflect(ray, bsdf, isect, renderer, scene, sample, rng)
			L += specular_transmit(ray, bsdf, isect, renderer, scene, sample, rng)

		return L



class LightStrategy(Enum):
	'''
	LightStrategy Class

	Enum for light strategy
	for direct lighting
	'''
	SAMPLE_ALL_UNIFORM = 1
	SAMPLE_ONE_UNIFORM = 2

class DirectLightingIntegrator(SurfaceIntegrator):
	'''
	DirectLightingIntegrator

	`SurfaceIntegrator` using direct lighting
	'''
	def __init__(self, strategy: 'LightStrategy'=LightStrategy.SAMPLE_ALL_UNIFORM, max_depth: INT=5):
		self.strategy = strategy
		self.max_depth = max_depth
		self.light_num_offset = 0
		self.light_sample_offsets = None
		self.bsdf_sample_offsets = None

	def __repr__(self):
		return "{}\nStrategy: {}\n".format(self.__class__, self.strategy)


	def request_samples(self, sampler: 'Sampler', sample: 'Sample', scene: 'Scene'):
		if self.strategy == LightStrategy.SAMPLE_ALL_UNIFORM:
			# sampling all lights
			n_lights = len(scene.lights)
			self.light_sample_offsets = []
			self.bsdf_sample_offsets = []
			for lgt in enumerate(scene.lights):
				n_smp = lgt.nSamples
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

	@jit
	def li(self, scene: 'Scene', renderer: 'Renderer', ray: 'RayDifferential',
			isect: 'Intersection', sample: 'Sample', rng=np.random.rand) -> 'Spectrum':
		L = Spectrum(0.)
		# Evaluate BSDF at hit point
		bsdf = isect.get_BSDF(ray)

		# Init
		p = bsdf.dgs.p
		n = bsdf.dgs.nn
		wo = -ray.d

		# compute emitted light if ray hit light src
		L += isec.le(wo)

		# compute direct lighting
		if len(scene.lights) > 0:
			if self.strategy == LightStrategy.SAMPLE_ALL_UNIFORM:
				L += uniform_sample_all_lights(scene, renderer, p, n, wo, isect.rEps, ray.time, 
						bsdf, sample, self.light_sample_offsets, self.bsdf_sample_offsets, rng)

			elif self.strategy == LightStrategy.SAMPLE_ONE_UNIFORM:
				L += uniform_sample_one_light(scene, renderer, p, n, wo, isect.rEps, ray.time, 
						bsdf, sample, self.light_num_offset, self.light_sample_offsets, self.bsdf_sample_offsets, rng)

			else:
				raise RuntimeError("Unknown LightStrategy")

		if ray.depth + 1 < self.max_depth:
			# trace rays for reflection and refraction
			L += specular_reflect(ray, bsdf, isect, renderer, scene, sample, rng)
			L += specular_transmit(ray, bsdf, isect, renderer, scene, sample, rng)

		return L



class PathIntegrator(SurfaceIntegrator):
	'''
	PathIntegrator

	Path tracing using Russian roulette.
	Also support maximum depth.
	'''
	SAMPLE_DEPTH = 3

	def __init__(self, max_depth: INT=5):
		self.max_depth = max_depth
		self.__light_num_offset = [0 for _ in PathIntegrator.SAMPLE_DEPTH]
		self.__light_sample_offsets = [None for _ in PathIntegrator.SAMPLE_DEPTH]
		self.__bsdf_sample_offsets = [None for _ in PathIntegrator.SAMPLE_DEPTH]
		self.__path_sample_offsets = [None for _ in PathIntegrator.SAMPLE_DEPTH]

	def request_samples(self, sampler: 'Sampler', sample: 'Sample', scene: 'Scene'):
		# after first few bounces switches to uniform random
		for i in range(self.SAMPLE_DEPTH):
			self.__light_sample_offsets[i] = LightSampleOffset(1, sample)
			self.__light_num_offset[i] = sample.add1d(1)
			self.__bsdf_sample_offsets[i] = BSDFSampleOffset(1, sample)
			self.__path_sample_offsets[i] = BSDFSampleOffset(1, sample)

	@jit
	def li(self, scene: 'Scene', renderer: 'Renderer', r: 'RayDifferential',
			isect: 'Intersection', sample: 'Sample', rng=np.random.rand) -> 'Spectrum':
		# common variables
		## product of BSDF and cosines for vertices
		## generated so far, divided by pdf's
		path_throughput = Spectrum(1.)

		## radiance from the running total of amount of
		## scattered ($\sum P(\bar{p_i})$)
		L = Spectrum(0.)

		## next ray to be traced to extend the path
		ray = r#RayDifferential.fromRD(r)

		## records if last outgoing direction sample
		## was due to specular reflection
		specular_bounce = False

		## most recently added vertex
		isectp = isect

		## subsequent vertex
		local_isect = Intersection()

		bounce_cnt = 0
		while True:
			# add possibly emitted light
			if bounce_cnt == 0 or specular_bounce:
				## emitted light is included by
				## previous tracing via direct lighting
				## exceptions for the first tracing or sampling a
				## specular direction since it is omitted from estimate_direct()
				L += path_throughput * isectp.le(-ray.d)

			# sample illumination from lights
			bsdf = isectp.get_BSDF(ray)
			p = bsdf.dgs.p
			n = bsdf.dgs.nn
			wo = -ray.d

			if bounce_cnt < PathIntegrator.SAMPLE_DEPTH:
				# use samples
				L += path_throughput * uniform_sample_one_light(scene, renderer, p, n, wo,
						isectp.rEps, ray.time, bsdf, sample, self.__light_num_offset[bounce_cnt],
						self.__light_sample_offsets[bounce_cnt], self.__bsdf_sample_offsets[bounce_cnt], rng=rng)
			else:
				# use uniform random
				L += path_throughput * uniform_sample_one_light(scene, renderer, p, n, wo,
						isectp.rEps, ray.time, bsdf, sample, rng=rng)

			# sample BSDF to get new direction
			## get BSDFSample for new direction
			if bounce_cnt < PathIntegrator.SAMPLE_DEPTH:
				out_bsdf_smp = BSDFSample.fromSample(sample, self.__path_sample_offsets[bounce_cnt], 0)
			else:
				out_bsdf_smp = BSDFSample.fromRand(rng)

			pdf, wi, flags, f = bsdf.sample_f(wo, out_bsdf_smp, BDFType.ALL)

			if f.is_black() or pdf == 0.:
				break

			specular_bounce = (flags & BDFType.SPECULAR) != 0
			path_throughput *= f * wi.abs_dot(n) / pdf
			ray = RayDifferential(p, wi, ray, isectp.rEps)

			# possibly terminate
			if bounce_cnt > PathIntegrator.SAMPLE_DEPTH:
				cont_prob = min(.5, path_throughput.y())	# high prob for terminating for low contribution paths
				if rng() > cont_prob:
					break

				# otherwise apply Russian roulette
				path_throughput /= cont_prob

			if bounce_cnt == self.max_depth:
				break

			# find next vertex
			hit, local_isect = scene.intersect(ray)
			if not hit:
				# ambient light
				if specular_bounce:
					for light in scene.lights:
						L += path_throughput * light.le(ray)
				break

			if bounce_cnt > 1:
				path_throughput *= renderer.transmittance(scene, ray, None, rng)

			isectp = local_isect

			bounce_cnt += 1

		return L













