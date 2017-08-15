"""
fresnel.py

pytracer.reflection package

Interface for Fresnel coefficients

Created by Jiayao on Aug 2, 2017
"""
from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
from pytracer import *
from pytracer.reflection.utility import (fr_cond, fr_diel)


# interface for Fresnel coefficients
class Fresnel(object, metaclass=ABCMeta):
	"""
	Fresnel Class

	Interface for computing Fresnel coefficients
	"""

	@abstractmethod
	def __call__(self, cosi: FLOAT):
		raise NotImplementedError('src.core.reflection.{}.__call__: abstract method '
									'called'.format(self.__class__))


class FresnelConductor(Fresnel):
	"""
	Fresnel Class

	Implement Fresnel interface for conductors
	"""
	def __init__(self, eta: 'Spectrum', k: 'Spectrum'):
		self.eta = eta
		self.k = k
	def __call__(self, cosi: FLOAT):
		return fr_cond(np.fabs(cosi), self.eta, self.k)


class FresnelDielectric(Fresnel):
	"""
	Fresnel Class

	Implement Fresnel interface for conductors
	"""
	def __init__(self, eta_i: 'Spectrum', eta_t: 'Spectrum'):
		self.eta_i = eta_i
		self.eta_t = eta_t

	def __call__(self, cosi: FLOAT):
		ci = np.clip(cosi, -1., 1.)
		# indices of refraction
		ei = self.eta_i
		et = self.eta_t
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
	"""
	FresnelFullReflect Class

	Return full `Spectrum` for all
	incoming directions
	"""
	def __call__(self, cosi: FLOAT):
		return Spectrum(1.)