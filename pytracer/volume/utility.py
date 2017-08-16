"""
utility.py


pytracer.volume package

Volume scattering utility functions.

Created by Jiayao on Aug 7, 2017
Modified on Aug 14, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo

__all__ = ['phase_isotrophic', 'phase_rayleigh', 'phase_mie_hazy',
           'phase_mie_murky', 'phase_hg', 'phase_schlick', 'subsurface_from_diffuse']


# Phase Functions
def phase_isotrophic(w: 'geo.Vector', wp: 'geo.Vector') -> FLOAT:
	return 1. / (4. * PI)


def phase_rayleigh(w: 'geo.Vector', wp: 'geo.Vector') -> FLOAT:
	ct = w.dot(wp)
	return 3. / (16. * PI) * (1. + ct * ct)


def phase_mie_hazy(w: 'geo.Vector', wp: 'geo.Vector') -> FLOAT:
	ct = w.dot(wp)
	return (.5 + 4.5 * np.power(.5 * (1. + ct), 8.)) / (4. * PI)


def phase_mie_murky(w: 'geo.Vector', wp: 'geo.Vector') -> FLOAT:
	ct = w.dot(wp)
	return (.5 + 16.5 * np.power(0.5 * (1. + ct), 32.)) / (4. * PI);


def phase_hg(w: 'geo.Vector', wp: 'geo.Vector', g: FLOAT) -> FLOAT:
	"""
	g: asymmetry parameter
	controls the distribution of light
	"""
	ct = w.dot(wp)
	return ((1. - g * g) / np.power(1. + g * g - 2. * g * ct, 1.5)) / (4. * PI);


def phase_schlick(w: 'geo.Vector', wp: 'geo.Vector', g: FLOAT) -> FLOAT:
	k = 1.55 * g - .55 * g * g * g
	kct = k * w.dot(wp)
	return ((1. - k * k) / (1. - kct) * (1. - kct)) / (4. * PI);


# Utility Functions
def subsurface_from_diffuse(kd: 'Spectrum', mfp: FLOAT, eta: FLOAT) -> ['Spectrum', 'Spectrum']:
	"""
	Subsurface from Diffuse
	Returns:
	[sigma_a: 'Spectrum', sigma_prime_s: 'Spectrum']
	TODO
	"""
	pass


# Measured Data
def get_volume_scattering(name: str) -> ['Spectrum', 'Spectrum']:
	"""
	Load volume scattering data,
	returns `Spectrum`s of
	[sigma_a, sigma'_s]
	"""
	from pytracer.data.volume import MEASURED_SUF_SC
	if name in MEASURED_SUF_SC:
		return [Spectrum.from_rgb(MEASURED_SUF_SC[name][0]),
		        Spectrum.from_rgb(MEASURED_SUF_SC[name][1])]
	return [None, None]



