"""
bdf.py

pytracer.reflection.bdf package

Model bidirectional distribution functions.
	- BDF
	- BSDF

Created by Jiayao on Aug 2, 2017
"""
from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
from pytracer import *
import pytracer.geometry as geo
import pytracer.montecarlo as mc
from pytracer.reflection.utility import (abs_cos_theta, cos_theta, sin_theta_sq)
from pytracer.reflection.fresnel import FresnelDielectric

__all__ = ['BDFType', 'BDF', 'BRDF2BTDF', 'ScaledBDF',
           'SpecularReflection', 'SpecularTransmission', 'Lambertian']

class BDFType():
	"""
	Wrapper for
	integer enumeration with
	bitmask ops
	"""

	def __init__(self, v):
		if isinstance(v, BDFType):
			self.v = v.v
		else:
			self.v = UINT(v)  # raise exception if needed

	def __repr__(self):
		return "{}\nEnum: {}".format(self.__class__, self.v)

	def __invert__(self):
		return BDFType(~self.v)

	def __or__(self, other):
		return BDFType(self.v | other.v) if isinstance(other, BDFType) \
			else BDFType(self.v | UINT(other))

	def __and__(self, other):
		return BDFType(self.v & other.v) if isinstance(other, BDFType) \
			else BDFType(self.v & UINT(other))

	def __xor__(self, other):
		return BDFType(self.v ^ other.v) if isinstance(other, BDFType) \
			else BDFType(self.v ^ UINT(other))

	def __lshift__(self, other):
		return BDFType(self.v << other)

	def __rshift__(self, other):
		return BDFType(self.v >> other)

	REFLECTION = (1 << 0)
	TRANSMISSION = (1 << 1)
	DIFFUSE = (1 << 2)
	GLOSSY = (1 << 3)
	SPECULAR = (1 << 4)
	ALL_TYPES = (DIFFUSE | GLOSSY | SPECULAR)
	ALL_REFLECTION = (REFLECTION | \
	                  ALL_TYPES)
	ALL_TRANSMISSION = (TRANSMISSION | \
	                    ALL_TYPES)
	ALL = (ALL_REFLECTION | ALL_TRANSMISSION)


class BDF(object, metaclass=ABCMeta):
	"""
	BDF Class

	Models bidirectional distribution function.
	Base class for `BSDF` and `BRDF`
	"""

	def __init__(self, t: 'BDFType'):
		self.type = BDFType(t)

	def __repr__(self):
		return "{}\nType: {}".format(self.__class__, self.type.v)

	@abstractmethod
	def f(self, wo: 'geo.Vector', wi: 'geo.Vector') -> 'Spectrum':
		"""
		Returns the BDF for given pair of directions
		Asssumes light at different wavelengths are
		decoupled.
		"""
		raise NotImplementedError('src.core.reflection.{}.f(): abstract method '
		                          'called'.format(self.__class__))

	def pdf(self, wo: 'geo.Vector', wi: 'geo.Vector') -> FLOAT:
		"""
		pdf()

		Returns pdf of given direction
		"""
		if wo.z * wi.z > 0.:
			# same hemisphere
			return abs_cos_theta(wi) * INV_PI
		return 0.

	def sample_f(self, wo: 'geo.Vector', u1: FLOAT,
	             u2: FLOAT) -> [FLOAT, 'geo.Vector', 'Spectrum']:
		"""
		Handles scattering discribed by delta functions
		or random sample directions.
		Returns the spectrum, incident vector and pdf used in MC sampling.

		By default samples from a hemeisphere with
		cosine-wighted distribution.

		Returns:
		[pdf, wi, Spectrum]
		"""
		# cosine sampling
		wi = mc.cosine_sample_hemisphere(u1, u2)
		if wo.z < 0.:
			wi.z *= -1.

		return [self.pdf(wo, wi), self.f(wo, wi), wi]

	def rho_hd(self, w: 'geo.Vector', samples: [FLOAT]) -> 'Spectrum':
		"""
		Computes hemispherical-directional reflectance function.

		- w
			Incoming 'geo.Vector'
		- samples
			2d np array
		"""
		r = Spectrum(0.)
		for smp in samples:
			wi, pdf, f = self.sample_f(w, smp[0], smp[1])
			if pdf > 0.:
				r += f * abs_cos_theta(wi) / pdf

		r /= len(samples)
		return r

	def rho_hh(self, nSamples: INT, samples_1: [FLOAT], samples_2: [FLOAT]) -> 'Spectrum':
		"""
		Computs hemispherical-hemispherical reflectance function.

		- samples_1, samples_2
			2d np array
		"""
		r = Spectrum(0.)
		for i in range(nSamples):
			wo = mc.uniform_sample_hemisphere(samples_1[i][0], samples_1[i][1])
			pdf_o = INV_2PI

			pdf_i, wi, f = self.sample_f(wo, samples_2[i][0], samples_2[i][1])

			if pdf_i > 0.:
				r += f * abs_cos_theta(wi) * abs_cos_theta(wo) / (pdf_o * pdf_i)

		r /= (PI * nSamples)
		return r

	def match_flag(self, flag: 'BDFType') -> bool:
		return self.type.v & flag.v == self.type.v


# adapter from BRDF to BTDF
class BRDF2BTDF(BDF):
	"""
	BRDF2BTDF Class

	Adpater class to convert a BRDF to
	BTDF by flipping incident light direction
	and forward function calls to BRDF
	"""
	def __init__(self, b: 'BDF'):
		super().__init__(b.type ^ (BDFType.REFLECTION | BDFType.TRANSMISSION))
		self.brdf = b

	@staticmethod
	def switch(w: 'geo.Vector'):
		return geo.Vector(w.x, w.y, -w.z)

	# forward function calls
	def f(self, wo: 'geo.Vector', wi: 'geo.Vector'): return self.brdf.f(wo, self.switch(wi))

	def sample_f(self, wo: 'geo.Vector', u1: FLOAT,  u2: FLOAT):
		pdf, wi, f = self.brdf.sample_f(wo, u1, u2)
		return pdf, self.switch(wi), f

	def rho_hd(self, wo: 'geo.Vector', samples: [FLOAT]): return self.brdf.rho_hd(wo, samples)

	def rho_hh(self, nSamples: INT, samples_1: [FLOAT],
					samples_2: [FLOAT]): return self.brdf.rho_hh(nSamples, samples_1, samples_2)


# adapter for scaling BDF
class ScaledBDF(BDF):
	"""
	ScaledBDF Class

	Wrapper for scaling BDF based on
	given `Spectrum`
	"""
	def __init__(self, b: 'BDF', sc: 'Spectrum'):
		super().__init__(b.type)
		self.bdf = b
		self.s = sc

	# scale by spectrum
	def f(self, wo: 'geo.Vector', wi: 'geo.Vector'): return self.s * self.brdf.f(wo, wi)

	def sample_f(self, wo: 'geo.Vector', u1: FLOAT, u2: FLOAT):
		pdf, wi, f = self.brdf.sample_f(wo, u1, u2)
		return pdf, wi, self.s * f

	def rho_hd(self, wo: 'geo.Vector', samples: [FLOAT]): return self.s * self.brdf.rho_hd(wo, samples)

	def rho_hh(self, nSamples: INT, samples_1: [FLOAT],
					samples_2: [FLOAT]): return self.s * self.brdf.rho_hh(nSamples, samples_1, samples_2)


class SpecularReflection(BDF):
	"""
	SpecularReflection Class

	Models perfect specular reflection.
	"""

	def __init__(self, sp: 'Spectrum', fr: 'Fresnel'):
		super().__init__(BDFType.REFLECTION | BDFType.SPECULAR)
		self.R = sp
		self.fresnel = fr

	def f(self, wo: 'geo.Vector', wi: 'geo.Vector') -> 'Spectrum':
		"""
		Return 0., singularity at perfect reflection
		will be handled in light transport routines.
		"""
		return Spectrum(0.)

	def pdf(self, wo: 'geo.Vector', wi: 'geo.Vector') -> FLOAT:
		return 0.

	def sample_f(self, wo: 'geo.Vector', u1: FLOAT,
	             u2: FLOAT) -> [FLOAT, 'geo.Vector', 'Spectrum']:
		# find direction
		wi = geo.Vector(-wo.x, -wo.y, wo.z)

		return [1., wi, self.fresnel(cos_theta(wo)) * self.R / abs_cos_theta(wi)]  # 1. suggests no MC samples needed


class SpecularTransmission(BDF):
	"""
	SpecularTransmission Class

	Models specular transmission
	using delta functions
	"""

	def __init__(self, t: 'Spectrum', ei: FLOAT, et: FLOAT):
		super().__init__(BDFType.TRANSMISSION | BDFType.SPECULAR)
		self.fresnel = FresnelDielectric(ei, et)  # conductors do not transmit light
		self.T = t
		self.ei = ei
		self.et = et

	def f(self, wo: 'geo.Vector', wi: 'geo.Vector') -> 'Spectrum':
		"""
		Return 0., singularity at perfect reflection
		will be handled in light transport routines.
		"""
		return Spectrum(0.)

	def pdf(self, wo: 'geo.Vector', wi: 'geo.Vector') -> FLOAT:
		return 0.

	def sample_f(self, wo: 'geo.Vector', u1: FLOAT,
	             u2: FLOAT) -> [FLOAT, 'geo.Vector', 'Spectrum']:
		# find eta pair
		ei, et = self.ei, self.et
		if cos_theta(wo) > 0.:
			ei, et = et, ei

		# compute transmited ray direction
		si_sq = sin_theta_sq(wo)
		eta = ei / et
		st_sq = eta * eta * si_sq

		if st_sq >= 1.:
			return [0., None, None]

		ct = np.sqrt(max(0., 1. - st_sq))
		if cos_theta(wo) > 0.:
			ct = -ct
		wi = geo.Vector(eta * (-wo.x), eta * (-wo.y), ct)

		F = self.fresnel(cos_theta(wo.t))

		return [1., wi, (et * et) / (ei * ei) * (Spectrum(1.) - F) * \
		        self.T / abs_cos_theta(wi)]  # 1. suggests no MC samples needed


class Lambertian(BDF):
	"""
	Lambertian Class

	Models lambertian.
	"""

	def __init__(self, r: 'Spectrum'):
		"""
		R: Spectrum Reflectance
		"""
		super().__init__(BDFType(BDFType.REFLECTION | BDFType.DIFFUSE))
		if isinstance(r, Spectrum):
			self.R = r
		else:
			self.R = Spectrum(r)

	def f(self, wo: 'geo.Vector', wi: 'geo.Vector') -> 'Spectrum':
		"""
		Return 0., singularity at perfect reflection
		will be handled in light transport routines.
		"""
		return INV_PI * self.R

	def rho_hd(self, wo: 'geo.Vector', samples: [FLOAT]) -> 'Spectrum':
		return self.R

	def rho_hh(self, nSamples: INT, samples_1: [FLOAT], samples_2: [FLOAT]) -> 'Spectrum':
		return self.R

