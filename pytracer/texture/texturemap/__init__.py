"""
__init__.py


pytracer.texture.texturemap package

Texture map definitions.

Created by Jiayao on Aug 5, 2017
Modified on Aug 14, 2017
"""
from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
from pytracer import *
import pytracer.geometry as geo
import pytracer.transform as trans

__all__ = ['TextureMapping2D', 'TextureMapping3D', 'SphericalMapping2D',
           'CylindricalMapping2D', 'PlannarMapping2D', 'IdentityMapping3D']


class TextureMapping2D(object, metaclass=ABCMeta):
	def __repr__(self):
		return "{}".format(self.__class__)

	@abstractmethod
	def __call__(self, dg: 'geo.DifferentialGeometry') -> [FLOAT]:
		"""
		Mapping maps the point given by dg to
		(s, t) texture coordinates.
		Returning a list of `FLOAT`s:
		[s, t, dsdx, dtdx, dsdy, dtdy]
		"""
		raise NotImplementedError('src.core.texture.{}.map(): abstract method '
		                          'called'.format(self.__class__))


class UVMapping2D(TextureMapping2D):
	def __init__(self, su: FLOAT, sv: FLOAT, du: FLOAT, dv: FLOAT):
		self.su = su
		self.sv = sv
		self.du = du
		self.dv = dv

	def __call__(self, dg: 'geo.DifferentialGeometry') -> [FLOAT]:
		s = self.su * dg.u + self.du
		t = self.sv * dg.v + self.dv
		dsdx = self.su * dg.dudx
		dtdx = self.sv * dg.dvdx
		dsdy = self.su * dg.dudy
		dtdy = self.sv * dg.dvdy
		return [s, t, dsdx, dtdx, dsdy, dtdy]


class SphericalMapping2D(TextureMapping2D):
	def __init__(self, w2t: 'trans.Transform'):
		self.w2t = w2t

	def __sphere(self, p: 'geo.Point') -> [FLOAT]:
		"""
		Spherical Mapping for single
		point. Returns list
		[s, t].
		"""
		v = geo.normalize(self.w2t(p) - geo.Point(0., 0., 0.))
		theta = geo.spherical_theta(v)
		phi = geo.spherical_phi(v)
		return [theta * INV_PI, phi * INV_2PI]

	def __call__(self, dg: 'geo.DifferentialGeometry') -> [FLOAT]:
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
	def __init__(self, w2t: 'trans.Transform'):
		self.w2t = w2t

	def __cylinder(self, p: 'geo.Point') -> [FLOAT]:
		"""
		Cylinderical Mapping for single
		point. Returns list
		[s, t].
		"""
		v = geo.normalize(self.w2t(p) - geo.Point(0., 0., 0.))
		return [(PI + self.arctan2(v.y, v.x)) * INV_2PI, v.z]

	def __call__(self, dg: 'geo.DifferentialGeometry') -> [FLOAT]:
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
	def __init__(self, vs: 'geo.Vector', vt: 'geo.Vector', ds: FLOAT = 0., dt: FLOAT = 0.):
		self.vs = vs
		self.vt = vt
		self.ds = ds
		self.dt = dt

	def __call__(self, dg: 'geo.DifferentialGeometry') -> [FLOAT]:
		v = dg.p - geo.Point(0., 0., 0.)
		return [self.ds + v.dot(self.vs),
		        self.dt + v.dot(self.vt),
		        dg.dpdx.dot(self.vs),
		        dg.dpdx.dot(self.vt),
		        dg.dpdy.dot(self.vs),
		        dg.dpdy.dot(self.vt)]


class TextureMapping3D(object, metaclass=ABCMeta):
	"""
	TextureMapping3D Class

	Base class for 3D texture mappings
	"""

	def __repr__(self):
		return "{}".format(self.__class__)

	@abstractmethod
	def __call__(self, dg: 'geo.DifferentialGeometry') -> ['geo.Point', 'geo.Vector', 'geo.Vector']:
		"""
		Mapping 3D point to texture
		Returns a list:
		[p, dpdx, dpdy]
		where p is the mapped point, dpdx, dpdy
		are mapped derivatives.
		"""
		raise NotImplementedError('src.core.texture.{}.map(): abstract method '
		                          'called'.format(self.__class__))


class IdentityMapping3D(TextureMapping3D):
	def __init__(self, w2t: 'trans.Transform'):
		self.w2t = w2t

	def __call__(self, dg: 'geo.DifferentialGeometry') -> ['geo.Point', 'geo.Vector', 'geo.Vector']:
		return [self.w2t(dg.p), self.w2t(dg.dpdx), self.w2t(dg.dpdy)]

