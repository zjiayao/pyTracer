'''
reflection.py

Model distribution functions.
Convention:
	Incident light and viewing direction
	are normalized and face outwards;
	Normal faces outwards and is not
	flipped to lie in the same side as
	viewing direction.

Created by Jiayao on Aug 2, 2017
'''
from numba import jit
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
from src.core.pytracer import *
from src.core.geometry import *
from src.core.spectrum import *

# inline functions
def cos_theta(w: 'Vector'): return w.z
def abs_cos_theta(w: 'Vector'): return np.fabs(w.z)
def sin_theta_sq(w: 'Vector'): return max(0., 1. - w.z * w.z)
def sin_theta(w: 'Vector'): return np.sqrt(max(0., 1. - w.z * w.z))
@jit
def cos_phi(w: 'Vector'): return 1. if max(0., 1. - w.z * w.z) == 0. \
								 else np.clip(w.x / np.sqrt(max(0., 1. - w.z * w.z), -1., 1.)
@jit
def sin_phi(w: 'Vector'): return 0. if max(0., 1. - w.z * w.z) == 0. \
								 else np.clip(w.y / np.sqrt(max(0., 1. - w.z * w.z), -1., 1.)

# utility functions
@jit
def fr_diel(cos_i: FLOAT, cos_t: FLOAT, eta_i: 'Spectrum', eta_t: 'Spectrum') -> 'Spectrum':
	'''
	fr_diel

	Compute reflectance using Fresnel formula for
	dielectric materials and circularly polarized light
	'''
	r_para = ((eta_t * cos_i ) - (eta_i * cos_t)) / \
				((eta_t * cos_i) + (eta_t * cos_t))
	r_perp = ((eta_i * cos_i) - (eta_t * cos_t)) / \
				((eta_i * cos_i) + (eta_t * cos_t))
	return .5 * (r_para * r_para + r_perp * r_perp)

@jit
def fr_cond(cos_i: FLOAT, eta: 'Spectrum', k: 'Spectrum') -> 'Spectrum':
	'''
	fr_cond

	Compute approx. reflectance using Fresnel formula for
	conductor materials and circularly polarized light
	'''
	tmp = (eta * eta + k * k) * cos_i * cos_i
	r_para_sq = (tmp - (2. * eta * cos_i) + 1.) / \
				(tmp + (2. * eta * cos_i) + 1.)
	tmp = eta * eta + k * k
	r_perp_sq = (tmp - (2. * eta * cos_i) + cos_i * cos_i) / \
				(tmp + (2. * eta * cos_i) + cos_i * cos_i)
	return .5 * (r_para_sq + r_perp_sq)

# interface for Fresnel coefficients
class Fresnel(object):
	'''
	Fresnel Class

	Interface for computing Fresnel coefficients
	'''
	__metaclass__ = ABCMeta

	@abstractmethod
	def __call__(self, cosi: FLOAT):
		raise NotImplementedError('src.core.reflection.{}.__call__: abstract method '
									'called'.format(self.__class__))

class FresnelConductor(Fresnel):
	'''
	Fresnel Class

	Implement Fresnel interface for conductors
	'''
	def __init__(self, eta: 'Spectrum', k: 'Spectrum'):
		self.eta = eta
		self.k = k
	def __call__(self, cosi: FLOAT):
		return fr_cond(np.fabs(cosi), self.eta, self.k)

class FresnelDielectric(Fresnel):
	'''
	Fresnel Class

	Implement Fresnel interface for conductors
	'''
	def __init__(self, eta_i: 'Spectrum', eta_t: 'Spectrum'):
		self.eta_i = eta_i
		self.eta_t = eta_t

	def __call__(self, cosi: FLOAT):
		ci = np.clip(cosi, -1., 1.)
		# indices of refraction
		ei = self.eta_i
		et = selt.eta_t
		if cosi < 0.:
			# ray is on the inside
			ei, et = et, ei

		# Snell's law
		st = ei / et * np.sqrt(max(0., 1. - ci * ci))

		if st >= 1.:
			# total internal reflection
			return 1.

		else:
			ct = np.sqrt(max(0., 1. - st * st))
			return fr_diel(np.fabs(ci), ct, ei, et)

class FresnelFullReflect(Fresnel):
	'''
	FresnelFullReflect Class

	Return full `Spectrum` for all
	incoming directions
	'''
	def __call__(self, cosi: FLOAT):
		return Spectrum(1.)

# interface for distribution functions

class BDFType(Enum):
	REFLECTION = 1 << 0
	TRANSMISSION = 1 << 1
	DIFFUSE = 1 << 2
	GLOSSY = 1 << 3
	SPECULAR = 1 << 4
	ALL_TYPES = BDFType.DIFFUSE | \
				BDFType.GLOSSY | \
				BDFType.SPECULAR
	ALL_REFLECTION = BDFType.REFLECTION | \
					 BDFType.ALL_TYPES
	ALL_TRANSMISSION = BDFType.TRANSMISSION | \
					   BDFType.ALL_TRANSMISSION
	ALL = BDFType.ALL_REFLECTION | BDFType.ALL_TRANSMISSION

class BDF(object):
	'''
	BDF Class

	Models bidirectional distribution function.
	Base class for `BSDF` and `BRDF`
	'''
	__metaclass__ = ABCMeta	

	def __init__(self, t: 'BDFType'):
		self.type = t

	def __repr__(self):
		return "{}\nType: {}".format(self.__class__, self.type.name)


	@abstractmethod
	def f(self, wo: 'Vector', wi: 'Vector') -> 'Spectrum':
		'''
		Returns the BDF for given pair of directions
		Asssumes light at different wavelengths are
		decoupled.
		'''
		raise NotImplementedError('src.core.reflection.{}.f(): abstract method '
									'called'.format(self.__class__)) 			

	@abstractmethod
	def sample_f(self, wo: 'Vector', wi: 'Vector', u1: FLOAT, 
							u2: FLOAT, pdf: [FLOAT]) -> 'Spectrum:
		'''
		Handles scattering discribed by delta functions
		or random sample directions
		'''
		raise NotImplementedError('src.core.reflection.{}.sample_f(): abstract method '
									'called'.format(self.__class__)) 	

	@abstractmethod
	def rho_hd(self, wo: 'Vector', nSamples: INT, samples: [FLOAT]) -> 'Spectrum':
		'''
		Computs hemispherical-directional reflectance function.

		'''
		raise NotImplementedError('src.core.reflection.{}.sample_f(): abstract method '
									'called'.format(self.__class__)) 
	@abstractmethod
	def rho_hh(self, nSamples: INT, samples_1: [FLOAT], samples_2: [FLOAT]) -> 'Spectrum':
		'''
		Computs hemispherical-hemispherical reflectance function.

		'''
		raise NotImplementedError('src.core.reflection.{}.sample_f(): abstract method '
									'called'.format(self.__class__)) 

	def match_flag(self, flag: 'BDFType') -> bool:
		return (self.type & flag) == type


# adapter from BRDF to BTDF
class BRDF2BTDF(BDF):
	'''
	BRDF2BTDF Class

	Adpater class to convert a BRDF to
	BTDF by flipping incident light direction
	and forward function calls to BRDF
	'''
	def __init__(self, b: 'BDF'):
		super().__init__(b.type ^ (BDFType.REFLECTION | BDFType.TRANSMISSION))
		self.brdf = b

	@staticmethod
	def switch(w: 'Vector'):
		return Vector(w.x, w.y, -w.z)

	# forward function calls
	def f(self, wo: 'Vector', wi: 'Vector'): return self.brdf.f(wo, self.switch(wi))	
	def sample_f(self, wo: 'Vector', wi: 'Vector', u1: FLOAT, 
					u2: FLOAT, pdf: [FLOAT]): return self.brdf.sample_f(wo, self.switch(wi),
															u1, u2, pdf)	
	def rho_hd(self, wo: 'Vector', nSamples: INT,
				samples: [FLOAT]): return self.brdf.rho_hd(wo, nSamples, samples)
	def rho_hh(self, nSamples: INT, samples_1: [FLOAT],
					samples_2: [FLOAT]): return self.brdf.rho_hh(nSamples, samples_1, samples_2)

# adapter for scaling BDF
class ScaledBDF(BDF):
	'''
	ScaledBDF Class

	Wrapper for scaling BDF based on
	given `Spectrum`
	'''
	def __init__(self, b: 'BDF', sc: 'Spectrum'):
		super().__init__(b.type)
		self.bdf = b
		self.s = sc

	# scale by spectrum
	def f(self, wo: 'Vector', wi: 'Vector'): return self.s * self.brdf.f(wo, wi)	
	def sample_f(self, wo: 'Vector', wi: 'Vector', u1: FLOAT, 
					u2: FLOAT, pdf: [FLOAT]): return self.s * self.brdf.sample_f(wo, wi,
															u1, u2, pdf)	
	def rho_hd(self, wo: 'Vector', nSamples: INT,
				samples: [FLOAT]): return self.s * self.brdf.rho_hd(wo, nSamples, samples)
	def rho_hh(self, nSamples: INT, samples_1: [FLOAT],
					samples_2: [FLOAT]): return self.s * self.brdf.rho_hh(nSamples, samples_1, samples_2)



