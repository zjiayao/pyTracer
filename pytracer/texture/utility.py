"""
utility.py


pytracer.texture package

Texture utility functions and noises.

Created by Jiayao on Aug 5, 2017
Modified on Aug 14, 2017
"""
from __future__ import absolute_import
from enum import Enum
from pytracer import *
import pytracer.geometry as geo


__all__ = ['Lanczos', 'smooth_step', 'FBm', 'turbulence', 'noise',
           'ImageWrap', 'MIPMap']


def Lanczos(x: FLOAT, tau: FLOAT = 2.):
	x = np.fabs(x)

	# add np.ndarray or list support
	if hasattr(x, '__iter__'):
		for i, xx in enumerate(x):
			if xx < EPS:
				x[i] = 1.
			elif xx > 1.:
				x[i] = 0.
			else:
				x[i] = np.sinc(xx * tau) * np.sinc(xx)
		return np.array(x)

	if x < EPS:
		return 1.
	elif x > 1.:
		return 0.
	return np.sinc(x * tau) * np.sinc(x)


def smooth_step(lo: FLOAT, hi: FLOAT, val: FLOAT):
	"""
	smooth_step()

	Clip but with smooth interpolation
	"""
	v = np.clip(val, lo, hi)
	return v * v * (3. - 2. * v)


def FBm(p: 'geo.Point', dpdx: 'geo.Vector', dpdy: 'geo.Vector', omega: FLOAT, max_octaves: INT):
	"""
	FBm()

	Implements Fractional Brownian Motion
	"""
	# number of octaves
	s2 = max(dpdx.sq_length(), dpdy.sq_length())
	foct = min(max_octaves, 1. - .5 * np.log2(s2))
	octaves = util.ftoi(foct)

	# sum of octaves of noise
	s = 0.
	l = 1.
	o = 1.
	for i in range(octaves):
		sum += o * noise(l * p)
		l *= 1.99  # not 2, since noise() returns 0. at integral grids
		o *= omega
	par = foct - octaves
	s += o * smooth_step(.3, .7, par) * noise(l * p)

	return s


def turbulence(p: 'geo.Point', dpdx: 'geo.Vector', dpdy: 'geo.Vector', omega: FLOAT, max_octaves: INT):
	"""
	turbulence()

	L1 sum of noise terms
	"""
	# number of octaves
	s2 = max(dpdx.sq_length(), dpdy.sq_length())
	foct = min(max_octaves, 1. - .5 * np.log2(s2))
	octaves = util.ftoi(foct)

	# sum of octaves
	s = 0.
	l = 1.
	o = 1.
	for i in range(octaves):
		s += o * np.fabs(noise(l * p))
		l *= 1.99  # not 2, since noise() returns 0. at integral grids
		o *= omega
	par = foct - octaves
	s += o * smooth_step(.3, .7, par) * np.fabs(noise(l * p))

	return s


# Noise Methods
def _grad(x: INT, y: INT, z: INT, dx, dy, dz):
	from pytracer.data.noise import NOISE_PERM
	h = NOISE_PERM[NOISE_PERM[NOISE_PERM[x] + y] + z]
	h &= INT(15)  # lowest 4 bits
	u = dx if h < 8 or h == 12 or h == 13 else dy
	v = dy if h < 4 or h == 12 or h == 13 else dz
	if h & 1:
		u = -1
	if h & 2:
		v = -v
	return u + v


#
# def noise_wt(t: FLOAT):
# 	return t * t * t (10. - 15. * t + 6. * t * t)

def noise(pnt: 'geo.Point') -> FLOAT:
	"""
	noise()

	Implementes Perlin's noise function, generate
	noise for given point in space.
	parameter:
		- pnt
			array-like
	"""
	from pytracer.data.noise import NOISE_PERM_SIZE
	# compute noise cell coord. and offsets
	xi = util.ftoi(pnt[0])
	yi = util.ftoi(pnt[1])
	zi = util.ftoi(pnt[2])
	dx = pnt[0] - xi
	dy = pnt[1] - yi
	dz = pnt[2] - zi

	# compute gradient wwights
	xi &= (NOISE_PERM_SIZE - 1)
	yi &= (NOISE_PERM_SIZE - 1)
	zi &= (NOISE_PERM_SIZE - 1)
	w000 = _grad(xi, yi, zi, dx, dy, dz)
	w100 = _grad(xi + 1, yi, zi, dx - 1, dy, dz)
	w010 = _grad(xi, yi + 1, zi, dx, dy - 1, dz)
	w110 = _grad(xi + 1, yi + 1, zi, dx - 1, dy - 1, dz)
	w001 = _grad(xi, yi, zi + 1, dx, dy, dz - 1)
	w101 = _grad(xi + 1, yi, zi + 1, dx - 1, dy, dz - 1)
	w011 = _grad(xi, yi + 1, zi + 1, dx, dy - 1, dz - 1)
	w111 = _grad(xi + 1, yi + 1, zi + 1, dx - 1, dy - 1, dz - 1)

	# compute trilinear interpolation of weights
	# smooth function to ensure cont. second and third derivative
	wx = dx * dx * dx * (10. - 15. * dx + 6. * dx * dx)
	wy = dy * dy * dy * (10. - 15. * dy + 6. * dy * dy)
	wz = dz * dz * dz * (10. - 15. * dz + 6. * dz * dz)
	x00 = util.lerp(wx, w000, w100)
	x10 = util.lerp(wx, w010, w110)
	x01 = util.lerp(wx, w001, w101)
	x11 = util.lerp(wx, w011, w111)
	y0 = util.lerp(wy, x00, x10)
	y1 = util.lerp(wz, x01, x11)
	return util.lerp(wz, y0, y1)


# MIP Map
class ImageWrap(Enum):
	REPEAT = 0
	BLACK = 1
	CLAMP = 2


class MIPMap(object):
	"""
	MIPMap Class
	"""

	class SampleWeight(object):
		"""
		Wrapper Class

		For resampling
		"""

		def __init__(self, pix: INT = 0, wt: [FLOAT] = 0.):
			self.pix = pix
			self.wt = wt

		def __repr__(self):
			return "{}\nPixel: {}\nWeights: {}".format(self.__class__, self.pix, self.wt)

	@staticmethod
	def resample(ores: UINT, nres: UINT) -> ['SampleWeight']:
		if ores > nres:
			RuntimeError('src.core.texture.{}.resample(): New resolution should '
			             'be greater than the old one, abort resampling.'.format(ores, nres))
		fil_width = 2.
		wts = np.zeros([nres, 5], dtype=FLOAT)

		for i in range(nres):
			# resampling weights for i-th pixel
			ctr = (i + .5) * ores / nres
			wts[i][0] = util.ftoi(ctr - fil_width + .5)  # +.5 is necessary for flooring
			wts[i][1:5] = Lanczos((wts[i][0] + np.arange(4) + .5 - ctr) / fil_width)

			# geo.normalize filter weights
			wts[i][1:5] /= np.sum(wts[i][1:5])

		return wts

	# class static
	weight_lut_size = 128
	weight_lut = None

	def __init__(self, typename: type, img: 'np.ndarray', trilinear: bool = False,
	             max_aniso: FLOAT = 8., wrap: 'ImageWrap' = ImageWrap.REPEAT):
		"""
		image is the np.ndarray of type `typename`
		"""
		self.typename = typename
		self.trilinear = trilinear
		self.max_aniso = max_aniso
		self.wrap = wrap

		sres, tres = np.shape(img)

		resampled = None

		# Note: the resampling can work fine, but
		# it is terribly slow. I plan to optimize
		# it later, posssibly declaring `Spectrum`
		# related classes to `jitclass()`.
		if (not util.is_pow_2(sres)) or (not util.is_pow_2(tres)):
			# resample to power of 2
			sp = util.next_pow_2(sres)
			tp = util.next_pow_2(tres)
			# samping in s and t direction
			## s
			s_wts = MIPMap.resample(sres, sp)

			# init, take advantage of numba.jit
			resampled = np.empty([tp, sp], dtype=typename)
			for t in range(tres):
				for s in range(sp):
					resampled[t, s] = typename(0.)
					# (s, t) in s-zoomed image
					for j in range(4):
						orig_s = INT(s_wts[s][0] + j)
						if self.wrap == ImageWrap.REPEAT:
							orig_s = orig_s % sres
						elif self.wrap == ImageWrap.CLAMP:
							orig_s = np.clip(orig_s, 0, sres - 1)

						if orig_s >= 0 and orig_s < INT(sres):
							resampled[t, s] += s_wts[s][1 + j] * img[t, orig_s]
			## t
			t_wts = MIPMap.resample(tres, tp)
			t_tmp = np.zeros(tp, dtype=img.dtype)
			for s in range(sp):
				for t in range(tp):
					# (s, t) in s-zoomed image
					t_tmp[t] = self.typename(0.)
					for j in range(4):
						orig_t = INT(t_wts[t][0] + j)
						if self.wrap == ImageWrap.REPEAT:
							orig_t = INT(orig_t % tres)
						elif self.wrap == ImageWrap.CLAMP:
							orig_t = INT(np.clip(orig_t, 0, tres - 1))

						if orig_t >= 0 and orig_t < INT(tres):
							t_tmp[t] += t_wts[t][1 + j] * resampled[orig_t, s]

				resampled[:, s] = t_tmp.copy()
				if self.typename is Spectrum:
					for t in range(tp):
						resampled[t, s] = resampled[t, s].clip()

			sres, tres = sp, tp

		self.__width, self.__height = sres, tres

		# init levels of MIPMap
		self.__n_levels = 1 + INT(np.log2(max(sres, tres)))
		self.__pyramid = np.empty(self.__n_levels, dtype=self.typename)
		# most detailed level
		self.__pyramid[0] = img if resampled is None else resampled

		for i in range(1, self.__n_levels):
			# bottom-up init
			u, v = np.shape(self.__pyramid[i - 1])
			s_res = max(1, UINT(u // 2))
			t_res = max(1, UINT(v // 2))
			self.__pyramid[i] = np.empty([t_res, s_res])
			for t in range(t_res):
				for s in range(s_res):
					self.__pyramid[i][t, s] = \
						.25 * (self.texel(i - 1, 2 * s, 2 * t) +
						       self.texel(i - 1, 2 * s + 1, 2 * t) +
						       self.texel(i - 1, 2 * s, 2 * t + 1) +
						       self.texel(i - 1, 2 * s + 1, 2 * t + 1))

				# init EVA filter weighted if needed
		if MIPMap.wight_lut is None:
			alpha = 2.
			r2 = np.arange(MIPMap.weight_lut_size) / (MIPMap.weight_lut_size - 1)
			MIPMap.wight_lut = np.exp(-alpha * r2) - np.exp(-alpha)

		@property
		def texel(self, level: UINT, s: INT, t: INT):
			l = self.__pyramid[level]
			u, v = np.shape(l)
			# compute texel (s, t)
			if self.wrap == ImageWrap.REPEAT:
				s = s % u
				t = t % v
			elif self.wrap == ImageWrap.CLAMP:
				s = np.clip(s, 0, u - 1)
				t = np.clip(t, 0, v - 1)
			elif self.wrap == ImageWrap.BLACK:
				if s < 0 or s >= u or t < 0 or t >= v:
					return self.typename(0.)

			return l(s, t)

		@property
		def width(self):
			return self.__width

		@property
		def height(self):
			return self.__height

		@property
		def levels(self):
			return self.__n_levels

		def __triagle(self, level: UINT, s: FLOAT, t: FLOAT):
			level = np.clip(level, 0, self.__n_levels - 1)
			s, t = [s, t] * np.shape(self.__pyramid[level]) - .5
			s0, t0 = util.ftoi(s), util.ftoi(t)
			ds = s - s0
			dt = t - t0
			return (1. - ds) * (1. - dt) * self.texel(level, s0, t0) + \
			       (1. - ds) * dt * self.texel(level, s0, t0 + 1) + \
			       ds * (1. - dt) * self.texel(level, s0 + 1, t0) + \
			       ds * dt * self.texel(level, s0 + 1, t0 + 1)

		def __EWA(self, level: UINT, s, t, ds0, dt0, ds1, dt1):
			if level >= self.__n_levels:
				return self.texel(self.__n_levels - 1, 0, 0)
			# convert EWA coord to proper scale for level
			u, v = np.shape(self.__pyramid[level])
			s = s * u - .5
			t = t * v - .5
			ds0 *= u
			dt0 *= v
			ds1 *= u
			dt1 *= v

			# pnt inside ellipse:
			# $e(s, t) = A s ^ 2 + B s t + C t ^ 2 < F$
			A = dt0 * dt0 + dt1 * dt1 + 1
			B = -2. * (ds0 * dt0 + ds1 * dt1)
			C = ds0 * ds0 + ds1 * ds1 + 1
			# additional 1 ensures ellipse does not fall
			# into texels

			# compute ellipse coef to bound filtering region
			invF = 1. / (A * C - B * B * .25)
			A *= invF
			B *= invF
			C *= invF

			# compute ellipse's (s, t) bounding box
			det = -B * B + 4. * A * C
			invDet = 1. / det
			u_sq = np.sqrt(det * C)
			v_sq = np.srqt(det * A)
			s0 = util.ctoi(s - 2. * invDet * u_sq)
			s1 = util.ftoi(s + 2. * invDet * u_sq)
			t0 = util.ctoi(t - 2. * invDet * v_sq)
			t1 = util.ftoi(t + 2. * invDet * v_sq)

			# scan over bound and compute quad eqn.
			ret = self.typename(0.)
			sum_wt = 0.
			for ti in range(t0, t1 + 1):
				tt = ti - t
				for si in range(s0, s1 + 1):
					ss = si - s
					# compute squared radius and filter texel if inside
					r2 = A * ss * ss + B * ss * tt + C * tt * tt
					if r2 < 1.:
						# inside
						wt = MIPMap.wight_lut[min(r2 * MIPMap.weight_lut_size, MIPMap.weight_lut_size - 1)]
						ret += self.texel(level, si, ti) * wt
						sum_wt += wt

			return ret / sum_wt

		def look_up(self, param: [FLOAT]):
			"""
			Textel look-up

			Isotropic Triangle Filter:
			param := [s, t, width]

			EWA:
			param := [s, t, dsdx, dtdx, dsdy, dtdy]
			"""
			if len(param) == 2 or len(param) == 3:
				# Isotropic Triangle Filter
				# Chooses a level which filter covers
				# four texels
				if len(param) == 2:
					s, t = param
					width = 0.

				else:
					s, t, width = param

				# mipmap level
				level = self.__n_levels - 1 + np.log2(max(width, EPS))

				# trilinear interpolation
				# for smooth MIPMap transittion
				if level < 0:
					return self.__triangle(0, s, t)
				elif level >= self.__n_levels - 1:
					return self.__triangle(self.__n_levels, 0, 0)
				else:
					i_level = util.ftoi(level)
					delta = level - i_level
					return (1. - delta) * self.__triangle(i_level, s, t) + \
					       delta * self.__triangle(i_level + 1, s, t)

			elif len(param) == 6:
				# EWA
				s, t, ds0, dt0, ds1, dt1 = param
				if self.trilinear:
					return self.look_up([s, t, 2. * np.max(np.fabs(param[2:6]))])
				# compute ellipse minor and major axes
				if ds0 * ds0 + dt0 * dt0 < ds1 * ds1 + dt1 * dt1:
					ds0, ds1 = ds1, ds0
					dt0, dt1 = dt1, dt0

				major_len = np.sqrt(ds0 * ds0 + dt0 * dt0)
				minor_len = np.sqrt(ds1 * ds1 + dt1 * dt1)

				# clamp eccentricity if too large
				if minor_len * self.max_aniso < major_len and minor_len > 0.:
					scale = major_len / (minor_len * self.max_aniso)
					ds0 *= scale
					ds1 *= scale
					minor_len *= scale

				if minor_len == 0.:
					return self.__triangle(0, s, t)

				# choose level of detail and perform EWA filtering
				lod = max(0., self.__n_levels - 1. + np.log2(minor_len))
				lodi = util.ftoi(lod)

				d = lod - lodi

				return (1. - d) * self.__EWA(lodi, s, t, ds0, dt0, ds1, dt1) + \
				       d * self.__EWA(lodi + 1, s, t, ds0, dt0, ds1, dt1)
