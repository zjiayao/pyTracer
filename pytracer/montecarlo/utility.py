"""
utility.py

pytracer.montecarlo package

Monte Carlo utilities.

Created by Jiayao on Aug 8, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo

__all__ = ['balance_heuristic', 'power_heuristic', 'rejection_sample_disk',
           'uniform_sample_hemisphere', 'uniform_hemisphere_pdf', 'uniform_sample_sphere',
           'uniform_sphere_pdf', 'uniform_sample_disk', 'concentric_sample_disk',
           'cosine_sample_hemisphere', 'cosine_hemisphere_pdf', 'uniform_sample_cone',
           'uniform_cone_pdf', 'sample_hg', 'hg_pdf']


# Utility Functions
def balance_heuristic(nf: INT, fpdf: FLOAT, ng: int, gpdf: FLOAT) -> FLOAT:
	"""
	balance_heuristic()

	Using balance heruristic for two
	distributions.
	"""
	return (nf * fpdf) / (nf * fpdf + ng * gpdf)


def power_heuristic(nf: INT, fpdf: FLOAT, ng: INT, gpdf: FLOAT) -> FLOAT:
	"""
	power_heuristic()

	Using power heuristic for two
	distributions, with a power of 2
	"""
	f = nf * fpdf
	g = ng * gpdf
	return f * f / ( f * f + g * g )


def rejection_sample_disk(rng=np.random.rand) -> [FLOAT, FLOAT]:
	"""
	rejection_sample_disk()

	Rejection sampling
	"""
	x = 1. - 2. * rng()
	y = 1. - 2. * rng()
	while x * x + y * y > 1.:
		x = 1. - 2. * rng()
		y = 1. - 2. * rng()
	return [x, y]


def uniform_sample_hemisphere(u1: FLOAT, u2: FLOAT) -> 'geo.Vector':
	"""
	uniform_sample_hemisphere()

	Sampling unifromly
	on a unit hemishpere.
	`u1` and `u2` are two
	random numbers passed in.
	"""
	r = np.sqrt(max(0., 1. - u1 * u1))
	phi = 2. * PI * u2
	return geo.Vector(r * np.cos(phi), r * np.sin(phi), u1)


def uniform_hemisphere_pdf() -> FLOAT:
	"""
	uniform_hermisphere_pdf()

	Returns the pdf w.r.t. the
	solid angle
	"""
	return INV_2PI


def uniform_sample_sphere(u1: FLOAT, u2: FLOAT) -> 'geo.Vector':
	"""
	uniform_sample_sphere()

	Sampling unifromly
	on a unit shpere.
	`u1` and `u2` are two
	random numbers passed in.
	"""
	z = 1. - 2. * u1
	r = np.sqrt(max(0., 1. - z * z))
	phi = 2. * PI * u2
	return geo.Vector(r * np.cos(phi), r * np.sin(phi), z)


def uniform_sphere_pdf() -> FLOAT:
	"""
	uniform_sphere_pdf()

	Returns the pdf w.r.t. the
	solid angle
	"""
	return 1. / (4. * PI)


def uniform_sample_disk(u1: FLOAT, u2: FLOAT) -> [FLOAT, FLOAT]:
	"""
	uniform_sample_disk()

	Sampling unifromly
	on a unit disk.
	`u1` and `u2` are two
	random numbers passed in.
	Returns Cartasian coord. in the
	object coord. system.
	"""
	return [np.sqrt(u1) * np.cos(2. * PI * u2), np.sqrt(u1) * np.sin(2. * PI * u2)]


def concentric_sample_disk(u1: FLOAT, u2: FLOAT) -> [FLOAT, FLOAT]:
	"""
	concentric_sample_disk()

	Concentric sampling on
	unit disk by Peter Shirley.
	`u1` and `u2` are two
	random numbers passed in.
	Returns Cartasian coord. in the
	object coord. system.
	"""
	# mapping random numbers
	sx = 2. * u1 - 1.
	sy = 2. * u2 - 1.

	# center
	if sx == 0. and sy == 0.:
		return [0., 0.]

	# map square to (r, \theta)
	#   \2/
	#  3 * 1
	#   /4\
	if sx > -sy:
		if sx > sy:
			# 1
			r = sx
			theta = sy / sx
			if sy <= 0.:
				theta += 8.

		else:
			# 2
			r = sy
			theta = 2. - sx / sy
	else:
		if sx > sy:
			# 4
			r = -sy
			theta = 6. + sx / sy
		else:
			# 3
			r = -sx
			theta = 4. - sy / sx
	theta *= PI / 4.

	return [r * np.cos(theta), r * np.sin(theta)]


def cosine_sample_hemisphere(u1: FLOAT, u2: FLOAT) -> 'geo.Vector':
	"""
	cosine_sample_hemisphere()

	Sampling from a cosine-weighted
	hemishpere distribution by projecting
	concentric disk random samples
	vertically.
	`u1` and `u2` are two
	random numbers passed in.
	"""
	vec = concentric_sample_disk(u1, u2)
	vec.z = np.sqrt(max(0., 1. - vec.x * vec.x - vec.y * vec.y))
	return vec


def cosine_hemisphere_pdf(costheta: FLOAT, phi: FLOAT) -> FLOAT:
	"""
	cosine_hemisphere_pdf()

	Returns the pdf w.r.t.
	the solid angle of cosine_sample_hemisphere()
	"""
	return costheta * INV_PI


def uniform_sample_triangle(u1: FLOAT, u2: FLOAT) -> [FLOAT, FLOAT]:
	"""
	uniform_sample_triangle()

	Sampling uniformly from a triangle.
	`u1` and `u2` are two
	random numbers passed in.
	Returns the baricentric coord [u, v]
	"""
	return [1. - np.sqrt(u1), u2 * np.sqrt(u1)]


def uniform_sample_cone(u1: FLOAT, u2: FLOAT, ct_max: FLOAT,
			x: 'geo.Vector'=None, y: 'geo.Vector'=None, z: 'geo.Vector'=None) -> 'geo.Vector':
	"""
	uniform_sample_cone()

	Sample from a uniform distribution
	over the cone of directions.
	"""
	if x is None or y is None or z is None:
		ct = (1. - u1) + u1 * ct_max
		st = np.sqrt(1. - ct * ct)
		phi = u2 * 2. * PI
		return geo.Vector(np.scos(phi) * st, np.sin(phi) * st, ct)
	else:
		ct = util.lerp(u1, ct_max, 1.)
		st = np.sqrt(1. - ct * ct)
		phi = u2 * 2. * PI
		return np.cos(phi) * st * x + np.sin(phi) * st * y + ct * z


def uniform_cone_pdf(ct_max: FLOAT) -> FLOAT:
	"""
	uniform_cone_pdf()
	"""
	return 1. / (2. * PI * (1. - ct_max))


def sample_hg(w: 'geo.Vector', g: FLOAT, u1: FLOAT, u2: FLOAT) -> 'geo.Vector':
	"""
	sample_hg()

	Sampling from Henyey-Greestein
	phase function, i.e.,
	$$
	\cos(\theta) = \frac{1}{2g}\left(1 + g^2 - \left( \frac{1-g^2}{1-g+2g\zeta} \right) ^2 \right))
	$$
	"""
	if np.fabs(g) < EPS:
		ct = 1. - 2. * u1
	else:
		sqr = (1. - g * g) / (1. - g + 2. * g * u1)
		ct = (1. + g * g - sqr * sqr) / (2. * g)

	st = np.sqrt(max(0., 1 - ct * ct))
	phi = 2. * PI * u2
	_, v1, v2 = geo.coordinate_system(w)
	return geo.spherical_direction(st, ct, phi, v1, v2, w)


def hg_pdf(w: 'geo.Vector', wp: 'geo.Vector', g: FLOAT) -> FLOAT:
	"""
	hg_pdf()
	"""
	from pytracer.volume import phase_hg
	return phase_hg(w, wp, g)

