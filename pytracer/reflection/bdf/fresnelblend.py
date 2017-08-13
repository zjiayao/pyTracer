"""
fresnelblend.py

pytracer.reflection.bdf package

Fresnel Blend Model, Ashikhmin and Shirley
Account for, e.g., glossy on diffuse

Created by Jiayao on Aug 2, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo
import pytracer.montecarlo as mc
from pytracer.reflection.utility import abs_cos_theta
from pytracer.reflection.bdf.bdf import (BDFType, BDF)

__all__ = ['FresnelBlend']


class FresnelBlend(BDF):
	"""
	FresnelBlend Class

	Based on the weighted sum of
	glossy and diffuse term.
	"""

	def __init__(self, d: 'Spectrum', s: 'Spectrum', dist: 'MicrofacetDistribution'):
		super().__init__(BDFType.REFLECTION | BDFType.GLOSSY)
		self.Rd = d
		self.Rs = s
		self.distribution = dist

	def schlick(self, ct: FLOAT) -> 'Spectrum':
		"""
		Schlick (1992) Approximation
		of Fresnel reflection
		"""
		return self.Rs + np.power(1. - ct, 5.) * (Spectrum(1.) - self.Rs)

	def sample_f(self, wo: 'geo.Vector', u1: FLOAT,
	             u2: FLOAT) -> [FLOAT, 'geo.Vector', 'Spectrum']:
		if u1 < .5:
			u1 = 2. * u1
			# cosine sample the hemisphere
			wi = mc.cosine_sample_hemisphere(u1, u2)
			if wo.z < 0.:
				wi.z *= -1.
		else:
			u1 = 2. * (u1 - .5)
			pdf, wi = self.distribution.sample_f(wo, u1, u2)
			if wo.z * wi.z <= 0.:
				# not on the same hemisphere
				return [pdf, wi, Spectrum(0.)]
		return [self.pdf(wo, wi), wi, self.f(wo, wi)]

	def pdf(self, wo: 'geo.Vector', wi: 'geo.Vector') -> FLOAT:
		if wo.z * wi.z <= 0.:
			return 0.
		return .5 * (abs_cos_theta(wi) * INV_PI + self.distribution.pdf(wo, wi))

	def f(self, wo: 'geo.Vector', wi: 'geo.Vector') -> 'Spectrum':
		diffuse = (28. / (23. * PI)) * self.Rd * \
		          (Spectrum(1.) - self.Rs) * \
		          (1. - np.power(1. - .5 * abs_cos_theta(wi), 5)) * \
		          (1. - np.power(1. - .5 * abs_cos_theta(wo), 5))

		wh = geo.normalize(wi + wo)

		specular = self.distribution.D(wh) / \
		           (4. * wi.abs_dot(wh) * max(abs_cos_theta(wi), abs_cos_theta(wo))) * \
		           self.schlick(wi.dot(wh))

		return diffuse + specular

