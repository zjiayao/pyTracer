"""
utility.py

patracer.reflection package

Utility functions

Created by Jiayao on Aug 2, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo

__all__ = ['cos_theta', 'abs_cos_theta', 'sin_theta_sq', 'cos_phi',
           'sin_phi', 'fr_diel', 'fr_cond', 'brdf_remap']


# inline functions
def cos_theta(w: 'geo.Vector'):
	return w.z


def abs_cos_theta(w: 'geo.Vector'):
	return np.fabs(w.z)


def sin_theta_sq(w: 'geo.Vector'):
	return max(0., 1. - w.z * w.z)


def sin_theta(w: 'geo.Vector'):
	return np.sqrt(max(0., 1. - w.z * w.z))


def cos_phi(w: 'geo.Vector'):
	return 1. if max(0., 1. - w.z * w.z) == 0. \
		else np.clip(w.x / np.sqrt(max(0., 1. - w.z * w.z)), -1., 1.)


def sin_phi(w: 'geo.Vector'):
	return 0. if max(0., 1. - w.z * w.z) == 0. \
		else np.clip(w.y / np.sqrt(max(0., 1. - w.z * w.z)), -1., 1.)


# utility functions
def fr_diel(cos_i: FLOAT, cos_t: FLOAT, eta_i: 'Spectrum', eta_t: 'Spectrum') -> 'Spectrum':
	"""
	fr_diel

	Compute reflectance using Fresnel formula for
	dielectric materials and circularly polarized light
	"""
	r_para = ((eta_t * cos_i ) - (eta_i * cos_t)) / \
				((eta_t * cos_i) + (eta_t * cos_t))
	r_perp = ((eta_i * cos_i) - (eta_t * cos_t)) / \
				((eta_i * cos_i) + (eta_t * cos_t))
	return .5 * (r_para * r_para + r_perp * r_perp)


def fr_cond(cos_i: FLOAT, eta: 'Spectrum', k: 'Spectrum') -> 'Spectrum':
	"""
	fr_cond

	Compute approx. reflectance using Fresnel formula for
	conductor materials and circularly polarized light
	"""
	tmp = (eta * eta + k * k) * cos_i * cos_i
	r_para_sq = (tmp - (2. * eta * cos_i) + 1.) / \
				(tmp + (2. * eta * cos_i) + 1.)
	tmp = eta * eta + k * k
	r_perp_sq = (tmp - (2. * eta * cos_i) + cos_i * cos_i) / \
				(tmp + (2. * eta * cos_i) + cos_i * cos_i)
	return .5 * (r_para_sq + r_perp_sq)


def brdf_remap(wo: 'geo.Vector', wi: 'geo.Vector') -> 'geo.Point':
	"""
	Mapping regularly sampled BRDF
	using Marschner, 1998
	"""
	dphi = geo.spherical_phi(wi) - geo.spherical_phi(wo)
	if dphi < 0.:
		dphi += 2. * PI
	if dphi > 2. * PI:
		dphi -= 2. * PI
	if dphi > PI:
		dphi = 2. * PI - dphi

	return geo.Point(sin_theta(wi) * sin_theta(wo),
	                 dphi * INV_PI, cos_theta(wi) * cos_theta(wo))

