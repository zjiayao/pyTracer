"""
__init__.py


pytracer.texture.texture package

Texture interface and basic textures.

Generic interface.
NB: Instanciating types need supporting
copy() method.

Created by Jiayao on Aug 5, 2017
Modified on Aug 14, 2017
"""

from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
from enum import Enum
from pytracer import *
import pytracer.geometry as geo
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from pytracer.texture.texturemap import TextureMapping2D


class Texture(object, metaclass=ABCMeta):
	"""
	Texture Baseclass
	"""

	def __repr__(self):
		return "{}\n".format(self.__class__)

	@abstractmethod
	def __call__(self, dg: 'geo.DifferentialGeometry'):
		raise NotImplementedError('src.core.texture.{}.__call__(): abstract method '
							'called'.format(self.__class__))


class ConstantTexture(Texture):
	def __init__(self, value):
		self.value = value

	def __call__(self, dg: 'geo.DifferentialGeometry'):
		return self.value


class ScaleTexture(Texture):
	"""
	ScaleTexture Class

	Returns product of two textures' values.
	Ignoring antialiasing.
	"""
	def __init__(self, tex1: 'Texture', tex2: 'Texture'):
		self.tex1 = tex1
		self.tex2 = tex2

	def __call__(self, dg: 'geo.DifferentialGeometry'):
		return self.tex1(dg) * self.tex2(dg)


class MixTexture(Texture):
	"""
	MixTexture Class

	Linear interpolate two `Texture`s using
	`FLOAT` texture `amt`.

	T = tex1(dg) * (1. - amt(dg)) + tex2(dg) * amt(dg)
	Ignoring antialiasing.
	"""

	def __init__(self, tex1: 'Texture', tex2: 'Texture', amt: 'Texture'):
		self.tex1 = tex1
		self.tex2 = tex2
		self.amt = amt  # must be a texture returning `FLOAT`s

	def __call__(self, dg: 'geo.DifferentialGeometry'):
		t = self.amt(dg)
		return (1. - t) * self.tex1(dg) + t * self.tex2(dg)


class BilerpTexture(Texture):
	"""
	BilerpTexture Class

	Bilinear interpolation between four constants.
	I.e., four corners in (s, t) space, `v00`, `v01`,
	`v10`, `v11`.
	"""

	def __init__(self, mapping: 'TextureMapping2D', v00, v01, v10, v11):
		self.mapping = mapping
		self.v00 = v00  # same type as Texture type
		self.v01 = v01
		self.v10 = v10
		self.v11 = v11

	def __call__(self, dg: 'geo.DifferentialGeometry'):
		s, t, _, _, _, _ = self.mapping(dg)
		return (1. - s) * (1. - t) * self.v00 + (1. - s) * t * self.v01 + \
		       (1. - t) * s * self.v10 + s * t * self.v11


class UVTexture(Texture):
	"""
	UVTexture Class
	"""
	def __init__(self, mapping: 'TextureMapping2D'):
		self.mapping = mapping

	def __call__(self, dg: 'geo.DifferentialGeometry'):
		s, t, _, _, _, _ = self.mapping(dg)
		return Spectrum.from_rgb([s - util.ftoi(s), t - util.ftoi(t), 0.])


class Chekcerboard2DTexture(Texture):
	"""
	Chekcerboard2DTexture Class

	Checks are one unit wide in each direction.
	Alternating checks are shaded using passing
	`Texture`s.
	"""
	class aaMethod(Enum):
		NONE = 0
		CLOSEDFORM = 1

	def __init__(self, mapping: 'TextureMapping2D', tex1: 'Texture', tex2: 'Texture', aa: str='closedform'):
		self.mapping = mapping
		self.tex1 = tex1
		self.tex2 = tex2
		# aa specifies anti-aliasing method
		if aa is None or aa.lower() == 'none':
			self.method = Chekcerboard2DTexture.aaMethod.NONE
		elif aa.lower() == 'closedform':
			self.method = Chekcerboard2DTexture.aaMethod.CLOSEDFORM
		else:
			print('src.core.texture.{}.__init__(): unsupported antialiasing method '
					'{}, using CLOSEDFORM'.format(self.__class__, aa))
			self.method = Chekcerboard2DTexture.aaMethod.CLOSEDFORM


	def __call__(self, dg: 'geo.DifferentialGeometry'):
		s, t, dsdx, dtdx, dsdy, dtdy = self.mapping(dg)
		if self.method == Chekcerboard2DTexture.aaMethod.NONE:
			# point sample texture
			if (util.ftoi(s) + util.ftoi(t)) % 2 == 0:
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
			if util.ftoi(s0) == util.ftoi(s1) and util.ftoi(t0) == util.ftoi(t1):
				# inside
				if (util.ftoi(s) + util.ftoi(t)) % 2 == 0:
					return self.tex1(dg)
				return self.tex2(dg)

			## box filtering check region
			# util.ftoi(x / 2) + 2. * max(x / 2 - util.ftoi(x / 2) -.5, 0)
			si = ((util.ftoi(s1 / 2) + 2. * max(s1 / 2 - util.ftoi(s1 / 2) - .5, 0)) -
				  (util.ftoi(s0 / 2) + 2. * max(s0 / 2 - util.ftoi(s0 / 2) - .5, 0))) / \
				 (2. * ds)
			ti = ((util.ftoi(t1 / 2) + 2. * max(t1 / 2 - util.ftoi(t1 / 2) - .5, 0)) -
				  (util.ftoi(t0 / 2) + 2. * max(t0 / 2 - util.ftoi(t0 / 2) - .5, 0))) / \
				 (2. * dt)
			area_sq = si * ti - 2. * si * ti
			if ds > 1. or dt > 1.:
				area_sq = .5
			return (1. - area_sq * self.tex1(dg)) + area_sq * self.tex2(dg)


class Checkboard3DTexture(Texture):
	"""
	Chekcerboard3DTexture Class
	"""
	def __init__(self, mapping: 'TextureMapping2D', tex1: 'Texture', tex2: 'Texture'):
		self.mapping = mapping
		self.tex1 = tex1
		self.tex2 = tex2

	def __call__(self, dg: 'geo.DifferentialGeometry'):
		p, _, _ = self.mapping(dg)

		if (util.ftoi(p.x) + util.ftoi(p.y) + util.ftoi(p.z)) % 2 == 0:
			return self.tex1(dg)
		return self.tex2(dg)


class DotsTexture(Texture):
	"""
	DotsTexture Class

	Random Polka Dots
	"""
	def __init__(self, mapping: 'TextureMapping2D', inside: 'Texture',
					outside: 'Texture'):
		self.mapping = mapping
		self.inside_dot = inside
		self.outside_dot = outside

	def __call__(self, dg: 'geo.DifferentialGeometry'):
		# compute cell indices
		s, t, _, _, _, _ = self.mapping(dg)
		sc = util.ftoi(s + .5)
		tc = util.ftoi(t + .5)

		from pytracer.texture.utility import noise

		# return insidedot result if inside
		if noise(sc + .5 , tc + .5) > 0. :
			rad = .35
			max_shift = .5 - rad
			s_ctr = sc + max_shift * noise(sc + 1.5, tc + 2.8)
			t_ctr = tc + max_shift * noise(sc + 4.5, tc + 9.8)
			ds = s - s_ctr
			dt = t - t_ctr
			if ds * ds + dt * dt < rad * rad:
				# inside
				self.inside_dot(dg)


		return self.outside_dot(dg)


class FBmTexture(Texture):
	"""
	FBmTexture Class

	Bump mapping using fractional Brownian motion.
	"""
	def __init__(self, octaves: INT, roughness: FLOAT, mapping: 'TextureMapping3D'):
		self.omega = roughness
		self.octaves = octaves
		self.mapping = mapping

	def __call__(self, dg: 'geo.DifferentialGeometry'):
		from pytracer.texture.utility import FBm
		p, dpdx, dpdy = self.mapping(dg)
		return FBm(p, dpdx, dpdy, self.omega, self.octaves)


class WrinkledTexture(Texture):
	"""
	WrinkledTexture Class

	Bump mapping using turbulence().
	"""
	def __init__(self, octaves: INT, roughness: FLOAT, mapping: 'TextureMapping3D'):
		self.omega = roughness
		self.octaves = octaves
		self.mapping = mapping

	def __call__(self, dg: 'geo.DifferentialGeometry'):
		p, dpdx, dpdy = self.mapping(dg)
		from pytracer.texture.utility import turbulence
		return turbulence(p, dpdx, dpdy, self.omega, self.octaves)


class WindyTexture(Texture):
	"""
	WindyTexture Class

	Two calls to fractional Brownian motion functions:
	1. low frequency variations over the surface (wind strength)
	2. amplitude of the wave at point (independent of wind)
	"""
	def __init__(self, mapping: 'TextureMapping3D'):
		self.mapping = mapping

	def __call__(self, dg: 'geo.DifferentialGeometry'):
		from pytracer.texture.utility import FBm
		p, dpdx, dpdy = self.mapping(dg)
		wind = FBm(.1 * p, .1 * dpdx, .1 * dpdy, .5, 3)
		wave = FBm(p, dpdx, dpdy, .5, 6)
		return np.fabs(wind) * wave


class MarbleTexture(Texture):
	"""
	MarbleTexture Class

	Used for perturbing texture coordinates before
	using another `Texture`.
	"""
	spline = [  [ .58, .58, .6 ], [ .58, .58, .6 ], [ .58, .58, .6 ],
				[ .5, .5, .5 ], [ .6, .59, .58 ], [ .58, .58, .6 ],
				[ .58, .58, .6 ], [.2, .2, .33 ], [ .58, .58, .6 ], ]
	spline_num = INT(27)
	def __init__(self, octaves: INT, roughness: FLOAT, scale: FLOAT, var: FLOAT, mapping: 'TextureMapping3D'):
		self.octaves = octaves
		self.omega = roughness
		self.scale = scale
		self.var = var
		self.mapping = mapping

	def __call__(self, dg: 'geo.DifferentialGeometry'):
		from pytracer.texture.utility import FBm
		p, dpdx, dpdy = self.mapping(dg)
		p *= self.scale
		marble = p.y + self.var * FBm(p, self.scale * dpdx, self.scale * dpdy, self.omega, self.octaves)
		wind = FBm(.1 * p, .1 * dpdx, .1 * dpdy, .5, 3)
		wave = FBm(p, dpdx, dpdy, .5, 6)
		t = .5 * .5 * np.sin(marble)

		# evaluate marble spline at $t$
		fst = util.ftoi(t * self.spline_num - 3)
		t = t * (self.spline_num - 3) - fst
		c0 = Spectrum.from_rgb(self.spline[fst:fst + 3])
		c1 = Spectrum.from_rgb(self.spline[fst + 1:fst + 4])
		c2 = Spectrum.from_rgb(self.spline[fst + 2:fst + 5])
		c3 = Spectrum.from_rgb(self.spline[fst + 3:fst + 6])

		# Bezier spline evaluated with de Castilejau's algorithm
		s0 = (1. - t) * c0 + t * c1
		s1 = (1. - t) * c1 + t * c2
		s2 = (1. - t) * c2 + t * c3

		s0 = (1. - t) * s0 + t * s1
		s1 = (1. - t) * s1 + t * s2

		return 1.5 * ((1. - t) * s0 + t * s1)	# 1.5 to increase variation


from pytracer.texture.texture.image import ImageTexture

__all__ = ['Texture', 'ImageTexture', 'ConstantTexture', 'SpectrumType', 'MixTexture',
           'BilerpTexture', 'UVTexture', 'Chekcerboard2DTexture', 'Checkboard3DTexture',
           'DotsTexture', 'FBmTexture', 'WrinkledTexture', 'WindyTexture', 'MarbleTexture']

