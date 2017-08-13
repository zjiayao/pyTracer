"""
camera.py

The base class to model cameras.

Created by Jiayao on July 31, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
from pytracer import *
import pytracer.geometry as geo
import pytracer.transform as trans
import pytracer.sampler as spler
import pytracer.film as flm

__all__ = ['Camera']


class Camera(object, metaclass=ABCMeta):
	"""
	Camera Class
	"""
	def __init__(self, c2w: 'trans.Animatedtrans.Transform', s_open: FLOAT,
				 s_close: FLOAT, film: 'flm.Film'):
		self.c2w = c2w
		self.s_open = s_open
		self.s_close = s_close
		self.film = film

	def __repr__(self):
		return "{}\nShutter: {} - {}".format(self.__class__, self.s_open, self.s_close)

	@abstractmethod
	def generate_ray(self, sample: 'spler.CameraSample') -> [FLOAT, 'geo.Ray']:
		"""
		Generate ray based on image sample.
		Returned ray direction is normalized

		@param
			- sample: instance of `spler.CameraSample` class
		@return
			- FLOAT: light weight
			- geo.Ray: generated `geo.Ray` object
		"""
		raise NotImplementedError('src.core.camera.{}.generate_ray: abstract method called' \
								  .format(self.__class__))

	def generate_ray_differential(self, sample: 'spler.CameraSample') -> [FLOAT, 'geo.RayDifferential']:
		"""
		Generate ray differential.
		"""
		wt, rd = self.generate_ray(sample)
		rd = geo.RayDifferential.fromgeo.Ray(rd)

		# find ray shift along x
		xshift = spler.CameraSample.from_sample(sample)
		xshift.imageX += 1
		wtx, rx = self.generate_ray(xshift)
		rd.rxOrigin = rx.o
		rd.rxDirection = rx.d

		# find ray shift along y
		yshift = spler.CameraSample.from_sample(sample)
		yshift.imageY += 1
		wty, ry = self.generate_ray(yshift)
		rd.ryOrigin = ry.o
		rd.ryDirection = ry.d

		if wtx == 0. or wty == 0.:
			return [0., rd]

		rd.hasDifferentials = True
		return [wt, rd]

