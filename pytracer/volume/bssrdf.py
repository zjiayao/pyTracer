"""
bssrdf.py


pytracer.volume package

Models BSSRDF

Created by Jiayao on Aug 7, 2017
Modified on Aug 14, 2017
"""
from __future__ import absolute_import
from pytracer import *

__all__ = ['BSSRDF']


class BSSRDF(object):
	"""
	BSSRDF Class

	Models low-level scattering properties.
	Scattering computed by integrators.
	"""
	def __init__(self, sig_a: 'Spectrum', sigp_s: 'Spectrum', e: FLOAT):
		self.__e = e
		self.__sig_a = sig_a
		self.__sigp_s = sigp_s #reduced scattering coefficient (1. - sigma_s)

	def __repr__(self):
		return "{}\nEta: {} sigma_a: {} sigma'_s: {}\n".format(self.__class__,
						self.__sig_a, self.__sigp_s)

	@property
	def eta(self):
		return self.__e

	@property
	def sigma_a(self):
		return self.__sig_a

	@property
	def sigma_prime_s(self):
		return self.__sigp_s