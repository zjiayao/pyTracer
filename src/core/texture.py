'''
texture.py

Model textures

Created by Jiayao on Aug 5, 2017
'''
from numba import jit
from abc import ABCMeta, abstractmethod
import numpy as np
from src.core.pytracer import *
from src.core.geometry import *
from src.core.diffgeom import *
from src.core.spectrum import *
from src.core.reflection import *

class TextureMapping2D(object, metaclass=ABCMeta):
	def __repr__(self):
		return "{}".format(self.__class__)

	@abstractmethod
	def map(self, dg: 'DifferentialGeometry') -> [FLOAT]:
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

	def map(self, dg: 'DifferentialGeometry') -> [FLOAT]:
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


	def map(self, dg: 'DifferentialGeometry') -> [FLOAT]:
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


	def map(self, dg: 'DifferentialGeometry') -> [FLOAT]:
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

	def map(self, dg: 'DifferentialGeometry') -> [FLOAT]:
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
	def map(self, dg: 'DifferentialGeometry') -> ['Point', 'Vector', 'Vector']:
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

	def map(self, dg: 'DifferentialGeometry') -> ['Point', 'Vector', 'Vector']:
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
		return "{}".format(self.__class__)

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
		return (1. - s) * (1. - t) * self.v00 + (1 - s.) * t * self.v01 + \
			   (1. - t) * s * self.v10 + s * t * self.v11
		


















