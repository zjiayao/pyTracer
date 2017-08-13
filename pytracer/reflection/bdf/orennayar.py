"""
orrennryer.py

pytracer.reflection.bdf package

Oren Nayer Model for Rough Surfaces


Created by Jiayao on Aug 2, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo
from pytracer.reflection.utility import (abs_cos_theta, sin_theta, cos_phi, sin_phi)
from pytracer.reflection.bdf.bdf import (BDFType, BDF)


# Oren Nayer Model for Rough Surfaces
class OrenNayar(BDF):
	"""
	OrenNayar Class

	Using Oren-Nayer approximation
	for surface scattering:
	$$
	f_r( \omega_i, \omega_o ) = \frac{\pho}{\pi} (A + B \max(0, \
		\cos(phi_i - \phi_o)) * \sin \alpha \tan \beta )
	$$
	where
	$$
	\alpha = \max(\theta_i, \theta_o) \\
	\beta = \min(\theta_i, \theta_o)
	$$
	"""

	def __init__(self, r: 'Spectrum', sig: FLOAT):
		super().__init__(BDFType.REFLECTION | BDFType.DIFFUSE)
		self.R = r
		sig = np.deg2rad(sig)
		sig_sq = sig * sig
		self.A = 1. - (sig_sq / (2. * (sig_sq + 0.33)))
		self.B = .45 * sig_sq / (sig_sq + .09)

	def f(self, wo: 'geo.Vector', wi: 'geo.Vector') -> 'Spectrum':
		st_i = sin_theta(wi)
		st_o = sin_theta(wo)

		# \cos(\phi_i - \phi_o)
		mc = 0.
		if st_i > EPS and st_o > EPS:
			dcos = cos_phi(wi) * cos_phi(wo) + sin_phi(wi) * sin_phi(wo)
			mc = max(0., dcos)

		# \sin \alpah \tan \beta
		if abs_cos_theta(wi) > abs_cos_theta(wo):
			sa = st_o
			tb = st_i / abs_cos_theta(wi)
		else:
			sa = st_i
			tb = st_o / abs_cos_theta(wo)

		return self.R * INV_PI * (self.A + self.B * mc * sa * tb)


