"""
environment.py

Implements environment camera


Created by Jiayao on July 31, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo
import pytracer.transform as trans
from pytracer.camera.camera import Camera

__all__ = ['EnvironmentCamera']


class EnvironmentCamera(Camera):
	"""
	EnvironmentCamera

	Models equirectangular projection of the scene
	"""

	def __init__(self, c2w: 'trans.AnimatedTransform', s_open: FLOAT, s_close: FLOAT, f: 'Film'):
		super().__init__(c2w, s_open, s_close, f)

	def generate_ray(self, sample: 'CameraSample') -> [FLOAT, 'geo.Ray']:
		"""
		Generate ray based on image sample.
		Returned ray direction is normalized
		"""
		time = util.lerp(sample.time, self.s_open, self.s_close)

		# compute ray direction
		theta = np.pi * sample.imageY / self.film.yResolution
		phi = 2 * np.pi * sample.imageX / self.film.xResolution
		stheta = np.sin(theta)

		ray = self.c2w(geo.Ray(geo.Point(0., 0., 0.),
				geo.Vector(stheta * np.cos(phi), np.cos(theta), stheta * np.sin(phi)), 0., np.inf, time))
		return [1., ray]
