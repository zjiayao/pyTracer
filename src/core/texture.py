'''
texture.py

Model textures

Created by Jiayao on Aug 5, 2017
'''
from numba import jit
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
from src.core.pytracer import *
from src.core.geometry import *
from src.core.diffgeom import *
from src.core.spectrum import *
from src.core.reflection import *


# Global Utility Functions
@jit
def Lanczos(x: FLOAT, tau: FLOAT=2.):
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

class TextureMapping2D(object, metaclass=ABCMeta):
	def __repr__(self):
		return "{}".format(self.__class__)

	@abstractmethod
	def __call__(self, dg: 'DifferentialGeometry') -> [FLOAT]:
		'''
		Mapping maps the point given by dg to
		(s, t) texture coordinates.
		Returning a list of `FLOAT`s:
		[s, t, dsdx, dtdx, dsdy, dtdy]
		'''
		raise NotImplementedError('src.core.texture.{}.map(): abstract method '
							'called'.format(self.__class__)) 	

class UVMapping2D(TextureMapping2D):
	def __init__(self, su: FLOAT, sv: FLOAT, du: FLOAT, dv: FLOAT):
		self.su = su
		self.sv = sv
		self.du = du
		self.dv = dv

	def __call__(self, dg: 'DifferentialGeometry') -> [FLOAT]:
		s = self.su * dg.u + self.du
		t = self.sv * dg.v + self.dv
		dsdx = self.su * dg.dudx
		dtdx = self.sv * dg.dvdx
		dsdy = self.su * dg.dudy
		dtdy = self.sv * dg.dvdy
		return [s, t, dsdx, dtdx, dsdy, dtdy]

class SphericalMapping2D(TextureMapping2D):
	def __init__(self, w2t: 'Transform'):
		self.w2t = w2t

	def __sphere(self, p: 'Point') -> [FLOAT]:
		'''
		Spherical Mapping for single
		point. Returns list
		[s, t].
		'''
		v = normalize(w2t(p) - Point(0., 0., 0.))
		theta = spherical_theta(v)
		phi = spherical_phi(v)
		return [theta * INV_PI, phi * INV_2PI]


	def __call__(self, dg: 'DifferentialGeometry') -> [FLOAT]:
		s, t = self.__sphere(dg.p)
		# compute texture coordinate
		# differentials
		# using forward differencing
		delta = .1

		sx, tx = self.__sphere(dg.p + delta * dg.dpdx)
		dsdx = (sx - s) / delta
		dtdx = (tx - t) / delta
		if dtdx > .5:
			dtdx = 1. - dtdx
		elif dtdx < -.5:
			dtdx = -(dtdx + 1.)

		sy, ty = self.__sphere(dg.p + delta * dg.dpdy)
		dsdy = (sy - s) / delta
		dtdy = (ty - s) / delta
		if dtdy > .5:
			dtdy = 1. - dtdy
		elif dtdy < -.5:
			dtdy = -(dtdy + 1.)

		return [s, t, dsdx, dtdx, dsdy, dtdy]


class CylindricalMapping2D(TextureMapping2D):
	def __init__(self, w2t: 'Transform'):
		self.w2t = w2t

	def __cylinder(self, p: 'Point') -> [FLOAT]:
		'''
		Cylinderical Mapping for single
		point. Returns list
		[s, t].
		'''
		v = normalize(w2t(p) - Point(0., 0., 0.))
		return [(PI + self.arctan2(v.y, v.x)) * INV_2PI, v.z]


	def __call__(self, dg: 'DifferentialGeometry') -> [FLOAT]:
		s, t = self.__cylinder(dg.p)
		# compute texture coordinate
		# differentials
		# using forward differencing
		delta = .1

		sx, tx = self.__cylinder(dg.p + delta * dg.dpdx)
		dsdx = (sx - s) / delta
		dtdx = (tx - t) / delta
		if dtdx > .5:
			dtdx = 1. - dtdx
		elif dtdx < -.5:
			dtdx = -(dtdx + 1.)

		sy, ty = self.__cylinder(dg.p + delta * dg.dpdy)
		dsdy = (sy - s) / delta
		dtdy = (ty - s) / delta
		if dtdy > .5:
			dtdy = 1. - dtdy
		elif dtdy < -.5:
			dtdy = -(dtdy + 1.)

		return [s, t, dsdx, dtdx, dsdy, dtdy]
		

class PlannarMapping2D(TextureMapping2D):
	def __init__(self, vs: 'Vector', vt: 'Vector', ds: FLOAT=0., dt: FLOAT=0.):
		self.vs = vs
		self.vt = vt
		self.ds = ds
		self.dt = dt

	def __call__(self, dg: 'DifferentialGeometry') -> [FLOAT]:
		v = dg.p - Point(0., 0., 0.)
		return [self.ds + v.dot(self.vs),
				self.dt + v.dot(self.vt),
				dg.dpdx.dot(self.vs),
				dg.dpdx.dot(self.vt),
				dg.dpdy.dot(self.vs),
				dg.dpdy.dot(self.vt)]

class TextureMapping3D(object, metaclass=ABCMeta):
	'''
	TextureMapping3D Class

	Base class for 3D texture mappings
	'''
	def __repr__(self):
		return "{}".format(self.__class__)

	@abstractmethod
	def __call__(self, dg: 'DifferentialGeometry') -> ['Point', 'Vector', 'Vector']:
		'''
		Mapping 3D point to texture
		Returns a list:
		[p, dpdx, dpdy]
		where p is the mapped point, dpdx, dpdy
		are mapped derivatives.
		'''
		raise NotImplementedError('src.core.texture.{}.map(): abstract method '
							'called'.format(self.__class__)) 	

class IdentityMapping3D(TextureMapping3D):
	def __init__(self, w2t: 'Transform'):
		self.w2t = w2t

	def __call__(self, dg: 'DifferentialGeometry') -> ['Point', 'Vector', 'Vector']:
		return [self.w2t(dg.p), self.w2t(dg.dpdx), self.w2t(dg.dpdy)]


# Texture Interface
# As a template, support various types, e.g.,
# Spectrum, FLOAT, &c
# NB: Instanciating types need supporting
# copy() method.
class Texture(object, metaclass=ABCMeta):
	'''
	Texture Baseclass
	'''

	def __repr__(self):
		return "{}\n".format(self.__class__)

	@abstractmethod
	def __call__(self, dg: 'DifferentialGeometry'):
		raise NotImplementedError('src.core.texture.{}.__call__(): abstract method '
							'called'.format(self.__class__)) 	


class ConstantTexture(Texture):
	def __init__(self, value):
		self.value = value.copy()

	def __call__(self, dg: 'DifferentialGeometry'):
		return self.value

class ScaleTexture(Texture):
	'''
	ScaleTexture Class

	Returns product of two textures' values.
	Ignoring antialiasing.
	'''
	def __init__(self, tex1: 'Texture', tex2: 'Texture'):
		self.tex1 = tex1
		self.tex2 = tex2

	def __call__(self, dg: 'DifferentialGeometry'):
		return self.tex1(dg) * self.tex2(dg)

class MixTexture(Texture):
	'''
	MixTexture Class

	Linear interpolate two `Texture`s using
	`FLOAT` texture `amt`.

	T = tex1(dg) * (1. - amt(dg)) + tex2(dg) * amt(dg)
	Ignoring antialiasing.
	'''
	def __init__(self, tex1: 'Texture', tex2: 'Texture', amt: 'Texture'):
		self.tex1 = tex1
		self.tex2 = tex2
		self.amt = amt # must be a texture returning `FLOAT`s

	def __call__(self, dg: 'DifferentialGeometry'):
		t = self.amt(dg)
		return (1. - t) * self.tex1(dg) + t * self.tex2(dg)

class BilerpTexture(Texture):
	'''
	BilerpTexture Class

	Bilinear interpolation between four constants.
	I.e., four corners in (s, t) space, `v00`, `v01`,
	`v10`, `v11`.
	'''
	def __init__(self, mapping: 'TextureMapping2D', v00, v01, v10, v11):
		self.mapping = mapping
		self.v00 = v00	# same type as Texture type
		self.v01 = v01
		self.v10 = v10
		self.v11 = v11

	def __call__(self, dg: 'DifferentialGeometry'):
		s, t, _, _, _, _ = self.mapping(dg)
		return (1. - s) * (1. - t) * self.v00 + (1. - s) * t * self.v01 + \
			   (1. - t) * s * self.v10 + s * t * self.v11
		


class ImageTexture(Texture):
	'''
	ImageTexture Class

	Types used in memory storage and returning
	may differ. Thus two type parameters must be passed
	at initiation, i.e., `type_ret` the returning type
	when evaluating the `Texture` and `type_mem` the type
	used to storing texture in memory.
	'''
	class TexInfo(object):
		'''
		Wrapper class for
		keying different image
		texures.
		'''
		def __init__(self, filename:str, trilinear: bool, max_aniso: FLOAT, wrap: 'ImageWrap',
						scale: FLOAT, gamma: FLOAT):
			self.filename = filename
			self.trilinear = trilinear
			self.max_aniso = max_aniso
			self.wrap = wrap
			self.scale = scale
			self.gamma = gamma

		def __repr__(self):
			return "{}\nFilename: {}\nTrilinear: {}\nMax Aniso: {}, Wrap: {}\nScale: {}, Gamma: {}" \
					.format(self.__class__, self.trilinear, self.max_aniso, self.wrap, self.scale,
								self.gamma)

		def __hash__(self):
			return hash( (self.filename, self.trilinear, self.max_aniso, self.wrap,
							self.scale, self.gamma))
		def __eq__(self, other):
			return  (self.filename == other.filename) and \
					(self.trilinear == other.trilinear) and \
					(self.max_aniso == other.max_aniso) and \
					(self.wrap == other.wrap) and \
					(self.scale == other.scale) and \
					(self.gamma == other.gamma)

		def __ne__(self, other):
			return not (self == other)

	# class static
	textures = {}

	def __init__(self, type_ret: type, type_mem: type, mapping: 'TextureMapping2D', filename: str, trilinear: bool,
					max_aniso: FLOAT, wrap: 'ImageWrap', scale: FLOAT, gamma: FLOAT):
		self.type_ret = type_ret
		self.type_mem = type_mem

		self.mapping = mapping
		self.mipmap = ImageTexture.get_texture(filename, trilinear, max_aniso,
						wrap, scale, gamma)

	def __call__(self, dg: 'DifferentialGeometry'):
		mem = self.mipmap.look_up(self.mapping(dg))
		return ImageTexture.__convert__out__(mem, type_ret)


	@staticmethod
	def __convert__in__(texel: object, type_mem: type, scale: FLOAT, gamma: FLOAT):
		'''
		Convert input texel to storage type.

		Performs gamma correction if needed.
		'''

		if isinstance(texel, 'RGBSpectrum') and \
				type_mem is RGBSpectrum:
			return (scale * texel) ** gamma
		elif isinstance(texel, 'RGBSpectrum') and \
				type_mem is (FLOAT or float):
			return (scale * texel.y()) ** gamma
		else:
			raise TypeError('src.core.texture.ImageTexture.__convert__in__: '
					'undefined convertion between {} and {}'.format(texel.__class__, type_mem))

	@staticmethod
	def __convert__out__(texel: object, type_ret: type):
		'''
		Convert looking-up texture value
		from storage type to returning type.
		'''

		if isinstance(texel, 'RGBSpectrum') and \
				type_ret is Spectrum:
			return Spectrum.fromRGB(texel.toRGB())
		elif (isinstance(texel, float) or isinstance(texel, FLOAT)) and \
				type_ret is (float or FLOAT):
			return texel
		else:
			raise TypeError('src.core.texture.ImageTexture.__convert__in__: '
					'undefined convertion between {} and {}'.format(texel.__class__, type_ret))			


	@staticmethod
	def __clear__cache__():
		ImageTexture.textures = {}




		
	def get_texture(self, filename: str, trilinear: bool, max_aniso: FLOAT,
						wrap: 'ImageWrap', scale: FLOAT, gamma: FLOAT) -> 'MIPMap':
		# look in the cache
		tex_info = TexInfo(filename, trilinear, max_aniso, wrap, scale, gamma)
		if tex_info in ImageTexture.textures:
			return ImageTexture.textures[tex_info]

		try:
			texels = read_image(filename)
			width, height = np.shape(texels)
		except:
			print('src.core.texture.{}.get_texture(): cannot process file {}, '
					'use default one-valued MIPMap'.format(self.__class__, filename))
			texels = None

		if texels is not None:
			if not isinstance(texels, type_mem):
				conv = np.empty([height, width], dtype=object)
				for t in range(height):
					for s in range(width):
						conv[t, s] = ImageTexture.__convert_in__(texels[t, s], self.type_mem, scale, gamma)

			ret = MIPMap(self.type_mem, conv, trilinear, max_aniso, wrap)

		else:
			# one-valued MIPMap
			val = [[self.power(scale, gamma)]]
			ret = MIPMap(self.type_mem, val)

		ImageTexture.textures[tex_info] = ret
		return ret

# MIP Map
class ImageWrap(Enum):
	REPEAT = 0
	BLACK = 1
	CLAMP = 2

class MIPMap(object):
	'''
	MIPMap Class
	'''
	class SampleWeight(object):
		'''
		Wrapper Class

		For resampling
		'''
		def __init__(self, pix: INT = 0, wt: [FLOAT] = 0.):
			self.pix = pix
			self.wt = wt
		def __repr__(self):
			return "{}\nPixel: {}\nWeights: {}".format(self.__class__, self.pix, self.wt)
	@staticmethod
	@jit
	def resample(ores: UINT, nres: UINT) -> ['SampleWeight']:
		if ores > nres:
			RuntimeError('src.core.texture.{}.resample(): New resolution should '
					'be greater than the old one, abort resampling.'.format(ores, nres))
		fil_width = 2.
		wts = np.zeros([nres, 5], dtype=FLOAT)		

		for i in range(nres):
			# resampling weights for i-th pixel
			ctr = (i + .5) * ores / nres
			wts[i][0] = ftoi(ctr - fil_width + .5) # +.5 is necessary for flooring
			wts[i][1:5] = Lanczos((wts[i][0] + np.arange(4) + .5 - ctr) / fil_width)
			
			# normalize filter weights
			wts[i][1:5] /= np.sum(wts[i][1:5])

		return wts

	# class static
	weight_lut_size = 128
	weight_lut = None

	@jit
	def __init__(self, typename: type, img: 'np.ndarray', trilinear: bool,
					max_aniso: FLOAT, wrap: 'ImageWrap'):
		'''
		image is the np.ndarray of type `typename`
		'''
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
		if (not is_pow_2(sres)) or (not is_pow_2(tres)):
			# resample to power of 2
			sp = next_pow_2(sres)
			tp = next_pow_2(tres)
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
							resampled[t, s] += s_wts[s][1+j] * img[t, orig_s]
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
							t_tmp[t] += t_wts[t][1+j] * resampled[orig_t, s]

				resampled[:, s] = t_tmp.copy()
				if self.typename is Spectrum:
					for t in range(tp):
						resampled[t, s] = resampled[t, s].clip()
			
			sres, tres = sp, tp

		self.__width, self.__height = sres, tres

		# init levels of MIPMap
		self.__n_levels = 1 + INT(np.log2(max(sres, tres)))
		self.__pyramid = np.empty(n_levels, dtype=self.typename)
		# most detailed level
		self.__pyramid[0] = img if resampled is None else resampled


		for i in range(1, n_levels):
			# bottom-up init
			u, v = np.shape(self.__pyramid[i-1])
			s_res = max(1, UINT(u // 2))
			t_res = max(1, UINT(v // 2))
			self.__pyramid[i] = np.empty([t_res, s_res])
			for t in range(t_res):
				for s in range(s_res):
					self.__pyramid[i][t, s] = \
						.25 * ( self.texel(i-1, 2 * s, 2 * t) + 
								self.texel(i-1, 2 * s + 1, 2 * t) + 
								self.texel(i-1, 2 * s, 2 * t + 1) + 
								self.texel(i-1, 2 * s + 1, 2 * t + 1)) 

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

		@jit
		def __triagle(self, level: UINT, s: FLOAT, t: FLAOT):
			level = np.clip(level, 0, self.__n_levels - 1)
			s, t = [s, t] * np.shape(self.__pyramid[level]) - .5
			s0, t0 = ftoi(s), ftoi(t)
			ds = s - s0
			dt = t - t0
			return  (1. - ds) * (1. - dt) * self.texel(level, s0, t0) + \
					(1. - ds) * dt 		  * self.texel(level, s0, t0 + 1) + \
					ds 		  * (1. - dt) * self.texel(level, s0 + 1, t0) + \
					ds 		  * dt 		  * self.texel(level, s0 + 1, t0 + 1)

		@jit
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
			s0 = ctoi(s - 2. * invDet * u_sq)
			s1 = ftoi(s + 2. * invDet * u_sq)
			t0 = ctoi(t - 2. * invDet * v_sq)
			t1 = ftoi(t + 2. * invDet * v_sq)

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

		@jit
		def look_up(self, param: [FLOAT]):
			'''
			Textel look-up

			Isotropic Triangle Filter:
			param := [s, t, width]

			EWA:
			param := [s, t, dsdx, dtdx, dsdy, dtdy]
			'''
			if len(param) == 3:
				# Isotropic Triangle Filter
				# Chooses a level which filter covers
				# four texels
				level, s, t = param

				# mipmap level
				level = self.__n_levels - 1 + np.log2(max(self.width, EPS))

				# trilinear interpolation
				# for smooth MIPMap transittion
				if level < 0:
					return self.__triangle(0, s, t)
				elif level >= self.__n_levels - 1:
					return self.__triangle(self.__n_levels, 0, 0)
				else:
					i_level = ftoi(level)
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
					ds1 *= scale
					ds2 *= scale
					minor_len *= scale

				if minor_len == 0.:
					return self.__triangle(0, s, t)

				# choose level of detail and perform EWA filtering
				lod = max(0., self.__n_levels - 1. + np.log2(minor_len))
				lodi = ftoi(lod)

				d = lod - lodi

				return (1. - d) * self.__EWA(lodi, s, t, ds0, dt0, ds1, dt1) + \
						d       * self.__EWA(lodi + 1, s, t, ds0, dt0, ds1, dt1)

# Procedural Texture
class UVTexture(Texture):
	'''
	UVTexture Class
	'''
	def __init__(self, mapping: 'TextureMapping2D'):
		self.mapping = mapping

	def __call__(self, dg: 'DifferentialGeometry'):
		s, t, _, _, _, _ = self.mapping(ds)
		return Spectrum.fromRGB([s - ftoi(s), t - ftoi(t), 0.])

class Chekcerboard2DTexture(Texture):
	'''
	Chekcerboard2DTexture Class

	Checks are one unit wide in each direction.
	Alternating checks are shaded using passing
	`Texture`s.
	'''
	class aaMethod(Enum):
		NONE = 0
		CLOSEDFORM = 1

	def __init__(self, mapping: 'TextureMapping2D', tex1: 'Texture', tex2: 'Texture', aa: str='closedform'):
		self.mapping = mapping
		self.tex1 = tex1
		self.tex2 = tex2
		# aa specifies anti-aliasing method
		if aa is None or aa.lower() == 'none':
			self.method = aaMethod.NONE
		elif aa.lower() == 'closedform':
			self.method = aaMethod.CLOSEDFORM
		else:
			print('src.core.texture.{}.__init__(): unsupported antialiasing method '
					'{}, using CLOSEDFORM'.format(self.__class__, aa))
			self.method = aaMethod.CLOSEDFORM


	def __call__(self, dg: 'DifferentialGeometry'):
		s, t, dsdx, dtdx, dsdy, dtdy = self.mapping(dg)
		if self.method == aaMethod.NONE:
			# point sample texture
			if (ftoi(s) + ftoi(t)) % 2 == 0):
				return self.tex1(dg)
			return self.tex2(dg)
		else:
			# compute closed-form box-filtered texture value
			## single check if filter inside
			ds = max(np.fabs(dsdx), np.fabs(dsdy))
			dt = max(np.fabs(dtdx), np.fabs(dtdy))
			s0 = s - ds
			s1 = s + ds
			t0 = t - dt
			t1 = t + dt
			if ftoi(s0) == ftoi(s1) and ftoi(t0) == ftoi(t1):
				# inside
				if (ftoi(s) + ftoi(t)) % 2 == 0):
					return self.tex1(dg)
				return self.tex2(dg)

			## box filtering check region
			ftoi(x / 2) + 2. * max(x / 2 - ftoi(x / 2) -.5, 0)
			si = ((ftoi(s1 / 2) + 2. * max(s1 / 2 - ftoi(s1 / 2) - .5, 0)) - 
				  (ftoi(s0 / 2) + 2. * max(s0 / 2 - ftoi(s0 / 2) - .5, 0))) / \
				 (2. * ds)
			ti = ((ftoi(t1 / 2) + 2. * max(t1 / 2 - ftoi(t1 / 2) - .5, 0)) - 
				  (ftoi(t0 / 2) + 2. * max(t0 / 2 - ftoi(t0 / 2) - .5, 0))) / \
				 (2. * dt)	
			area_sq = si * ti - 2. * si * ti
			if ds > 1. or dt > 1.:
				area_sq = .5
			return (1. - area_sq * tex1(dg)) + area_sq * tex2(dg)


class Checkboard3DTexture(Texture):
	'''
	Chekcerboard3DTexture Class
	'''
	def __init__(self, mapping: 'TextureMapping2D', tex1: 'Texture', tex2: 'Texture'):
		self.mapping = mapping
		self.tex1 = tex1
		self.tex2 = tex2

	def __call__(self, dg: 'DifferentialGeometry'):
		p, _, _ = self.mapping(dg)

		if (ftoi(p.x) + ftoi(p.y) + ftoi(p.z)) % 2 == 0:
			return self.tex1(dg)
		return self.tex2(dg)









