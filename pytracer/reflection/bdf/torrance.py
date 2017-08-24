"""
orrennryer.py

pytracer.reflection.bdf package

Torrance–Sparrow Model

Created by Jiayao on Aug 2, 2017
"""
from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
from pytracer import *
import pytracer.geometry as geo
from pytracer.reflection.utility import (abs_cos_theta, cos_theta, cos_phi, sin_phi)
from pytracer.reflection.bdf.bdf import (BDFType, BDF)

__all__ = ['MicrofacetDistribution', 'Blinn', 'Anisotropic', 'Microfacet']


class MicrofacetDistribution(object, metaclass=ABCMeta):
	"""
	MicrofacetDistribution Class

	Compute microfacet distribution using
	Torrance–Sparrow model.
	Microfacet distribution functions must
	be normalized.
	"""

	def __repr__(self):
		return "{}\n".format(self.__class__)

	@abstractmethod
	def D(self, wh: 'geo.Vector') -> FLOAT:
		raise NotImplementedError('src.core.reflection.{}.D(): abstract method '
		                          'called'.format(self.__class__))

	@abstractmethod
	def sample_f(self, wo: 'geo.Vector', u1: FLOAT,
	             u2: FLOAT) -> [FLOAT, 'geo.Vector', 'Spectrum']:
		raise NotImplementedError('src.core.reflection.{}.sample_f(): abstract method '
		                          'called'.format(self.__class__))

	@abstractmethod
	def pdf(self, wo: 'geo.Vector', wi: 'geo.Vector') -> FLOAT:
		raise NotImplementedError('src.core.reflection.{}.pdf(): abstract method '
		                          'called'.format(self.__class__))

	# Blinn Model


class Blinn(MicrofacetDistribution):
	"""
	Blinn Class

	Models Blinn microfacet distribution:
	$$
	D(\omega_h) \prop (\omega_h \cdot \mathbf(n)) ^ {e}
	$$
	Apply normalization constraint:
	$$
	D(\omega_h) = \frac{e+2}{2\pi} (\omega_h \cdot \mathbf{n}) ^ {e}
	$$
	"""

	def __init__(self, e: FLOAT):
		self.e = np.clip(e, -np.inf, 10000.)

	def D(self, wh: 'geo.Vector') -> FLOAT:
		return (self.e + 2) * INV_2PI * np.power(abs_cos_theta(wh), self.e)

	def sample_f(self, wo: 'geo.Vector', u1: FLOAT,
	             u2: FLOAT) -> [FLOAT, 'geo.Vector']:
		# compute sampled half-angle vector
		ct = np.power(u1, 1. / (self.e + 1))
		st = np.sqrt(max(0., 1. - ct * ct))
		phi = u2 * 2. * PI
		wh = geo.spherical_direction(st, ct, phi)

		if wo.z * wh.z <= 0.:
			wh *= -1.

		# incident direction by reflection
		wi = -wo + 2. * wo.dot(wh) * wh

		# pdf
		pdf = ((self.e + 1.) * np.power(ct, self.e)) / \
		      (2. * PI * 4. * wo.dot(wh))

		if wo.dot(wh) <= 0.:
			pdf = 0.

		return [pdf, wi]

	def pdf(self, wo: 'geo.Vector', wi: 'geo.Vector') -> FLOAT:
		wh = geo.normalize(wo + wi)
		ct = abs_cos_theta(wh)

		if wo.dot(wh) <= 0.:
			return 0.

		return ((self.e + 1.) * np.power(ct, self.e)) / (2. * PI * 4. * wo.dot(wh))

# Anisotropic, Ashikhmin and Shirley
class Anisotropic(MicrofacetDistribution):
	def __init__(self, ex: FLOAT, ey: FLOAT):
		self.ex = np.clip(ex, -np.inf, 10000.)
		self.ey = np.clip(ey, -np.inf, 10000.)

	def D(self, wh: 'geo.Vector') -> FLOAT:
		cth = abs_cos_theta(wh)
		d = 1. - cth * cth

		if d == 0.:
			return 0.

		e = (self.ex * wh.x * wh.x + self.ey * wh.y * wh.y) / d
		return np.sqrt((self.ex + 2.) * (self.ey + 2.)) * INV_2PI * np.power(cth, e)

	def __sample_first_quad(self, u1: FLOAT, u2: FLOAT) -> [FLOAT, FLOAT]:
		"""
		__sample_first_quad()

		Samples a direction in the first quadrant of
		unit hemisphere. Returns [phi, cos(theta)]
		"""
		if self.ex == self.ey:
			phi = PI * u1 * .5
		else:
			phi = np.arctan(np.sqrt((self.ex + 1.) / (self.ey + 1.)) * np.tan(PI * u1 * .5))

		cp = np.cos(phi)
		sp = np.sin(phi)

		return [phi, np.power(u2, 1. / (self.ex * cp * cp + self.ey * sp * sp + 1.))]

	def sample_f(self, wo: 'geo.Vector', u1: FLOAT,
	             u2: FLOAT) -> [FLOAT, 'geo.Vector']:
		# sample from first quadrant and remap to hemisphere to sample w_h
		if u1 < .25:
			phi, ct = self.__sample_first_quad(4. * u1, u2)
		elif u1 < .5:
			u1 = 4. * (.5 - u1)
			phi, ct = self.__sample_first_quad(u1, u2)
			phi = PI - phi
		elif u1 < .75:
			u1 = 4. * (u1 - .5)
			phi, ct = self.__sample_first_quad(u1, u2)
			phi += PI
		else:
			u1 = 4. * (1. - u1)
			phi, ct = self.__sample_first_quad(u1, u2)
			phi = 2. * PI - phi

		st = np.sqrt(max(0., 1. - ct * ct))
		wh = geo.spherical_direction(st, ct, phi)
		if wo.z * wh.z <= 0.:
			wh *= -1.

		# incident direction by reflection
		wi = -wo + 2. * wo.dot(wh) * wh

		# compute pdf for w_i
		ct = abs_cos_theta(wh)
		ds = 1. - ct * ct
		if ds > 0. and wo.dot(wh) > 0.:
			return [(np.sqrt((self.ex + 1.) * (self.ey + 1.)) * INV_2PI * np.power(ct,
			                                                                       (
			                                                                       self.ex * wh.x * wh.x + self.ey * wh.y * wh.y) / ds)) / \
			        (4. * wo.dot(wh)), wi]
		else:
			return [0., wi]

	def pdf(self, wo: 'geo.Vector', wi: 'geo.Vector') -> FLOAT:
		wh = geo.normalize(wo + wi)
		ct = abs_cos_theta(wh)
		ds = 1. - ct * ct
		if ds > 0. and wo.dot(wh) > 0.:
			return (np.sqrt((self.ex + 1.) * (self.ey + 1.)) * INV_2PI * np.power(ct,
			                                                                      (
			                                                                      self.ex * wh.x * wh.x + self.ey * wh.y * wh.y) / ds)) / \
			       (4. * wo.dot(wh))
		else:
			return 0.


class Microfacet(BDF):
	"""
	Microfacet Class

	Models microfaceted surface
	"""

	def __init__(self, r: 'Spectrum', f: 'Fresnel', d: 'MicrofacetDistribution'):
		super().__init__(BDFType(BDFType.REFLECTION | BDFType.GLOSSY))
		self.R = r
		self.fresnel = f
		self.distribution = d

	def f(self, wo: 'geo.Vector', wi: 'geo.Vector') -> 'Spectrum':
		ct_i = cos_theta(wi)
		ct_o = cos_theta(wo)

		if ct_i == 0. or ct_o == 0.:
			return Spectrum(0.)

		wh = geo.normalize(wi + wo)
		ct_h = wi.dot(wh)
		F = self.fresnel(ct_h)

		return self.R * self.distribution.D(wh) * self.G(wo, wi, wh) * \
		       F / (4. * ct_i * ct_o)

	def G(self, wo: 'geo.Vector', wi: 'geo.Vector', wh: 'geo.Vector') -> FLOAT:
		return min(1., min((2. * abs_cos_theta(wh) * abs_cos_theta(wo) / wo.abs_dot(wh)),
		                   (2. * abs_cos_theta(wh) * abs_cos_theta(wi) / wo.abs_dot(wh))))

	def sample_f(self, wo: 'geo.Vector', u1: FLOAT,
	             u2: FLOAT) -> [FLOAT, 'geo.Vector', 'Spectrum']:
		pdf, wi = self.distribution.sample_f(wo, u1, u2)
		if wi.z * wo.z <= 0.:
			return [pdf, wi, Spectrum(0.)]
		return [pdf, wi, self.f(wo, wi)]

	def pdf(self, wo: 'geo.Vector', wi: 'geo.Vector') -> FLOAT:
		if wi.z * wo.z <= 0.:
			return 0.
		return self.distribution.pdf(wo, wi)

