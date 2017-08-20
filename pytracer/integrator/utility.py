"""
utility.py

Utility functions for
integrators.

Created by Jiayao on Aug 9, 2017
"""

from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo


__all__ = ['compute_light_sampling_cdf', 'uniform_sample_all_lights',
           'uniform_sample_one_light', 'specular_reflect', 'specular_transmit', 'estimate_direct']


def compute_light_sampling_cdf(scene: 'Scene') -> 'Distribution1D':
	"""
	compute_light_sampling_cdf()

	Creates a one dimensional
	distribution based on the power
	of all lights in the scene.
	"""
	n_lights = len(scene.lights)
	power = []
	for light in scene.lights:
		power.append(light.power(scene).y())

	from pytracer.montecarlo import Distribution1D

	return Distribution1D(power)


def uniform_sample_all_lights(scene: 'Scene', renderer: 'Renderer', p: 'geo.Point', n: 'geo.Normal',
                              wo: 'geo.Vector', r_eps: FLOAT, time: FLOAT, bsdf: 'BSDF', sample: 'Sample', light_offsets: ['LightSampleOffset'],
                              bsdf_offsets: ['BSDFSampleOffset'], rng=np.random.rand):
	"""
	uniform_sample_all_lights()

	Estimate contribution by each light
	individually using MC.
	"""
	from pytracer.light import LightSample
	from pytracer.reflection import (BDFType, BSDFSample)
	
	L = Spectrum(0.)

	for i, light in enumerate(scene.lights):
		n_smp = 1 if light_offsets is None else light_offsets[i].nSamples
		# estimate direct lighting
		## find light and bsdf for estimating
		Ld = Spectrum(0.)
		for j in range(n_smp):
			if light_offsets is not None and bsdf_offsets is not None:
				light_smp = LightSample.from_sample(sample, light_offsets[i], j)
				bsdf_smp = BSDFSample.from_sample(sample, bsdf_offsets[i], j)
			else:
				light_smp = LightSample.from_rand(rng)
				bsdf_smp = BSDFSample.from_rand(rng)

			Ld += estimate_direct(scene, renderer, light, p, n, wo, r_eps, time, bsdf,
			                      light_smp, bsdf_smp, BDFType(BDFType.ALL & ~BDFType.SPECULAR), rng)

		L += Ld / n_smp

	return L


def uniform_sample_one_light(scene: 'Scene', renderer: 'Renderer', p: 'geo.Point', n: 'geo.Normal',
                             wo: 'geo.Vector', r_eps: FLOAT, time: FLOAT, bsdf: 'BSDF', sample: 'Sample', light_num_offset: INT=-1,
                             light_offsets: ['LightSampleOffset']=None, bsdf_offsets: ['BSDFSampleOffset']=None, rng=np.random.rand):
	"""
	uniform_sample_one_light()

	random choose one light to sample
	and multiply with number of lights
	to compensate on average
	"""
	from pytracer.light import LightSample
	from pytracer.reflection import (BDFType, BSDFSample)
	# choose a light
	# light power based importance sampling
	# is to implement
	n_lights = len(scene.lights)
	if n_lights == 0:
		return Spectrum(0.)

	if light_num_offset == -1:
		# use random
		light_num = util.ftoi(rng() * n_lights)
	else:
		light_num = util.ftoi(sample.oneD[light_num_offset][0] * n_lights)

	light_num = min(light_num, n_lights - 1)
	light = scene.lights[light_num]

	# init
	if light_offsets is not None and bsdf_offsets is not None:
		light_smp = LightSample.from_sample(sample, light_offsets[0], 0)
		bsdf_smp = BSDFSample.from_sample(sample, bsdf_offsets[0], 0)
	else:
		light_smp = LightSample.from_rand(rng)
		bsdf_smp = BSDFSample.from_rand(rng)

	return n_lights * estimate_direct(scene, renderer, light, p, n, wo, r_eps, time, bsdf,
	                                  light_smp, bsdf_smp, BDFType(BDFType.ALL & ~BDFType.SPECULAR), rng)


def specular_reflect(ray: 'geo.RayDifferential', bsdf: 'BSDF', isect: 'Intersection',
                     renderer: 'Renderer', scene: 'Scene', sample: 'Sample', rng) -> 'Spectrum':
	from pytracer.reflection import (BDFType, BSDFSample)

	wo = -ray.d
	p = bsdf.dgs.p

	n = bsdf.dgs.nn
	pdf, wi, _, f = bsdf.sample_f(wo, BSDFSample.from_rand(rng), BDFType(BDFType.REFLECTION | BDFType.SPECULAR))
	L = Spectrum(0.)
	if pdf > 0. and not f.is_black() and wi.abs_dot(n) != 0.:
		# compute ray differential
		rd = geo.RayDifferential(p, wi, ray, isect.rEps)
		if ray.has_differentials:
			rd.has_differentials = True
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

			rd.rxDirection = wi - dwodx + 2. * geo.Vector.fromgeo.Normal(wo.dot(n) * dndx + dDNdx * n)
			rd.ryDirection = wi - dwody + 2. * geo.Vector.fromgeo.Normal(wo.dot(n) * dndy + dDNdy * n)

		Li = renderer.li(scene, rd, sample, rng)
		L = f * Li * wi.abs_dot(n) / pdf

	return L


def specular_transmit(ray: 'geo.RayDifferential', bsdf: 'BSDF', isect: 'Intersection',
                      renderer: 'Renderer', scene: 'Scene', sample: 'Sample', rng) -> 'Spectrum':
	from pytracer.reflection import (BDFType, BSDFSample)
	wo = -ray.d
	p = bsdf.dgs.p
	n = bsdf.dgs.nn
	pdf, wi, _, f = bsdf.sample_f(wo, BSDFSample.from_rand(rng), BDFType(BDFType.TRANSMISSION | BDFType.SPECULAR))
	L = Spectrum(0.)
	if pdf > 0. and not f.is_black() and wi.abs_dot(n) != 0.:
		# compute ray differential
		rd = geo.RayDifferential(p, wi, ray, isect.rEps)
		if ray.has_differentials:
			rd.has_differentials = True
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

			rd.rxDirection = wi + eta * dwodx - geo.Vector.fromgeo.Normal(mu * dndx + dmudx * n)
			rd.ryDirection = wi + eta * dwody - geo.Vector.fromgeo.Normal(mu * dndy + dmudy * n)

		Li = renderer.li(scene, rd, sample, rng)
		L = f * Li * wi.abs_dot(n) / pdf

	return L


def estimate_direct(scene: 'Scene', renderer: 'Renderer', light: 'Light', p:'geo.Point', n: 'geo.Normal',
                    wo: 'geo.Vector', r_eps: FLOAT, time: FLOAT, bsdf: 'BSDF', light_smp: 'LightSample', bsdf_smp: 'BSDFSample',
                    flags: 'BDFType', rng=np.random.rand) -> 'Spectrum':
	"""
	estimate_direct()

	Estimate direct lighting
	using MC Multiple Importance Sampling.
	"""
	from pytracer.reflection import BDFType
	Ld = Spectrum(0.)
	# sample light source
	Li, wi, light_pdf, vis = light.sample_l(p, r_eps, light_smp, time)
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
				from pytracer.montecarlo import power_heuristic
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
				from pytracer.montecarlo import power_heuristic
				wt = power_heuristic(1, bsdf_pdf, 1, light_pdf)
				# add contribution
				Li = Spectrum(0.)
				ray = geo.RayDifferential(p, wi, r_eps, np.inf, time)
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

