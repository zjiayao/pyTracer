"""
surface.py

pytracer.integrator.surface package

Model surface integrators.

Created by Jiayao on Aug 9, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
from pytracer import *
import pytracer.geometry as geo
from pytracer.integrator.integrator import Integrator

__all__ = ['SurfaceIntegrator']


class SurfaceIntegrator(Integrator):
	"""
	SurfaceIntegrator Class

	Interface for surface integrators
	"""

	@abstractmethod
	def li(self, scene: 'Scene', renderer: 'Renderer', ray: 'geo.RayDifferential',
			isect: 'Intersection', sample: 'Sample', rng=np.random.rand) -> 'Spectrum':
		"""
		li()

		Returns the outgoing radiance at the intersection
		point of a given ray with scene.
		- scene
			`Scene` to be rendered
		- renderer
			`Renderer` used for rendering, `li()` or `transmittance()` might
			be called
		- ray
			`geo.Ray` to evaluate incident radiance
		- isect
			First `Intersection` of the ray in the `Scene`
		- sample
			A `Sample` generated by a `Sample` for this ray.
			Might be used for MC methods.
		- rng
			Random number generator, by default `numpy.random.rand`
		"""
		raise NotImplementedError('src.core.integrator.{}.li(): abstract method '
						'called'.format(self.__class__))

