"""
montecarlo.py

Monte Carlo utilities.

Created by Jiayao on Aug 8, 2017
"""

from src.geometry.diffgeom import *
from src.texture.texture import *



# Utility Functions

@jit
def balance_heuristic(nf: INT, fpdf: FLOAT, ng: int, gpdf: FLOAT) -> FLOAT:
	"""
	balance_heuristic()

	Using balance heruristic for two
	distributions.
	"""
	return (nf * fpdf) / (nf * fpdf + ng * gpdf)

@jit
def power_heuristic(nf: INT, fpdf: FLOAT, ng: INT, gpdf: FLOAT) -> FLOAT:
	"""
	power_heuristic()

	Using power heuristic for two
	distributions, with a power of 2
	"""
	f = nf * fpdf
	g = ng * gpdf
	return f * f / ( f * f + g * g )


@jit
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

@jit
def uniform_sample_hemisphere(u1: FLOAT, u2: FLOAT) -> 'Vector':
	"""
	uniform_sample_hemisphere()

	Sampling unifromly
	on a unit hemishpere.
	`u1` and `u2` are two
	random numbers passed in.
	"""
	r = np.sqrt(max(0., 1. - u1 * u1))
	phi = 2. * PI * u2
	return Vector(r * np.cos(phi), r * np.sin(phi), u1)

def uniform_hermisphere_pdf() -> FLOAT:
	"""
	uniform_hermisphere_pdf()

	Returns the pdf w.r.t. the
	solid angle
	"""
	return INV_2PI

@jit
def uniform_sample_sphere(u1: FLOAT, u2: FLOAT) -> 'Vector':
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
	return Vector(r * np.cos(phi), r * np.sin(phi), z)

def uniform_sphere_pdf() -> FLOAT:
	"""
	uniform_sphere_pdf()

	Returns the pdf w.r.t. the
	solid angle
	"""
	return 1. / (4. * PI)



@jit
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


@jit
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

@jit
def cosine_sample_hemisphere(u1: FLOAT, u2: FLOAT) -> 'Vector':
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


@jit
def uniform_sample_triangle(u1: FLOAT, u2: FLOAT) -> [FLOAT, FLOAT]:
	"""
	uniform_sample_triangle()

	Sampling uniformly from a triangle.
	`u1` and `u2` are two
	random numbers passed in.
	Returns the baricentric coord [u, v]
	"""
	return [1. - np.sqrt(u1), u2 * np.sqrt(u1)]

@jit
def uniform_sample_cone(u1: FLOAT, u2: FLOAT, ct_max: FLOAT,
			x: 'Vector'=None, y: 'Vector'=None, z: 'Vector'=None) -> 'Vector':
	"""
	uniform_sample_cone()

	Sample from a uniform distribution
	over the cone of directions.
	"""
	if x is None or y is None or z is None:
		ct = (1. - u1) + u1 * ct_max
		st = np.sqrt(1. - ct * ct)
		phi = u2 * 2. * PI
		return Vector(np.scos(phi) * st, np.sin(phi) * st, ct)
	else:
		ct = Lerp(u1, ct_max, 1.)
		st = np.sqrt(1. - ct * ct)
		phi = u2 * 2. * PI
		return np.cos(phi) * st * x + np.sin(phi) * st * y + ct * z


def uniform_cone_pdf(ct_max: FLOAT) -> FLOAT:
	"""
	uniform_cone_pdf()
	"""
	return 1. / (2. * PI * (1. - ct_max))

@jit
def sample_hg(w: 'Vector', g: FLOAT, u1: FLOAT, u2: FLOAT) -> 'Vector':
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
	_, v1, v2 = coordinate_system(w)
	return spherical_direction(st, ct, phi, v1, v2, w)

def hg_pdf(w: 'Vector', wp: 'Vector', g: FLOAT) -> FLOAT:
	"""
	hg_pdf()
	"""
	return phase_hg(w, wp, g)



# Utility Classes
class Distribution1D(object):
	"""
	Distribution1D Class

	Piecewise-constant 1D function's
	PDF and CDF and performs sampling.
	"""
	def __init__(self, arr: [FLOAT]):
		"""
		Noted len(arr) + 1 is needed
		for CDF.
		"""
		self.cnt = len(arr)
		self.func = arr.copy()
		self.cdf = np.empty(self.cnt+1, dtype=FLOAT)

		# compute integral of step function at $x_i$
		self.cdf[0] = 0.
		for i in range(1, self.cnt + 1):
			self.cdf[i] = self.cdf[i-1] + arr[i-1] / self.cnt

		# transform step function into CDF
		self.cdf_raw = self.cdf[-1]
		self.cdf /= self.cdf_raw

	def __repr__(self):
		return "{}\nSteps: {}\n".format(self.__class__, self.cnt)

	@jit
	def sample_cont(self, u: FLOAT) -> [FLOAT, FLOAT]:
		"""
		sample_cont()

		Use given random sample `u` to
		sample from its distribution.
		Returns the r.v. value and sampled
		pdf: [rv, pdf].
		"""
		# surrounding CDF segment
		idx = np.searchsorted(self.cdf, u) - 1
		off = max(0, idx)

		# compute offset
		du = (u - self.cdf[off]) / (self.cdf[off+1] - self.cdf[off])

		# compute pdf for sampled offset
		# and r.v. value
		return [(du + off) / self.cnt, self.func[off] / self.cdf_raw]

	@jit
	def sample_dis(self, u: FLOAT) -> [FLOAT, FLOAT]:
		"""
		sample_dis()

		Use given random sample `u` to
		sample from its distribution.
		Returns the r.v. value and sampled
		pdf: [rv, pdf].
		"""
		# surrounding CDF segment
		idx = np.searchsorted(self.cdf, u) - 1
		off = max(0, idx)

		return [off, self.func[off] / (self.cdf_raw * self.cnt)]

class Distribution2D(object):
	"""
	Distribution2D Class

	Piecewise-constant 2D function's
	PDF and CDF and performs sampling.
	"""
	def __init__(self, arr: 'np.ndarray'):
		"""
		arr is the sample values, with shape being
		n_v * n_u
		"""
		nv, nu = np.shape(arr)
		self.p_cond_v = []

		# conditional distribution p(u|v)
		for i in range(nv):
			self.p_cond_v.append(Distribution1D(arr[i, :]))
		
		# marginal distribution p(v)
		marginal_v = []
		for i in range(nv):
			marginal_v.append(self.p_cond_v[i].cdf_raw)
		self.p_marg_v = Distribution1D(marginal_v)


	def __repr__(self):
		return "{}\nSteps: {}\n".format(self.__class__, self.cnt)

	@jit
	def sample_cont(self, u0: FLOAT, u1: FLOAT) -> [list, FLOAT]:
		"""
		sample_cont()

		Use given random samples `u0`
		and `u1` to sample from distribution.

		First sampling from p(v) then from p(u|v).
		Returns [[u, v], pdf]
		"""
		v, pdf1 = self.p_marg_v.sample_cont(u1)
		v = np.clip(ftoi(v * self.p_marg_v.cnt), 0, self.p_marg_v.cnt - 1).astype(INT)

		u, pdf0 = self.p_cond_v[v].sample_cont(u0)

		return [[u, v], pdf0 * pdf1]

	@jit
	def pdf(self, u: FLOAT, v: FLOAT) -> FLOAT:
		"""
		pdf()

		Value of the pdf given a sample value
		"""
		ui = np.clip(ftoi(u * self.p_cond_v[0].cnt), 0, self.p_cond_v[0].cnt - 1).astype(INT)
		vi = np.clip(ftoi(v * self.p_marg_v.cnt), 0, self.p_marg_v.cnt - 1).astype(INT)

		if self.p_cond_v[vi].cdf_raw * self.p_marg_v.cdf_raw == 0.:
			return 0.

		return (self.p_cond_v[vi].func[ui] * self.p_marg_v.func[vi]) / \
					(self.p_cond_v[vi].cdf_raw * self.p_marg_v.cdf_raw)













