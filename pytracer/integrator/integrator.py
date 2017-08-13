"""
integrator.py

Model integrators.

Created by Jiayao on Aug 9, 2017
"""

from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
import pytracer.sampler as spler
import pytracer.scene as scn
import pytracer.renderer as ren
import pytracer.camera as cam

__all__ = ['Integrator']


# Integrator Interface
class Integrator(object, metaclass=ABCMeta):
	"""
	Integrator Class

	Base Class for Integrators
	"""
	def __repr__(self):
		return "{}\n".format(self.__class__)

	# optional preprocessing
	def preprocess(self, scene: 'scn.Scene', camera: 'cam.Camera', renderer:'ren.Renderer'):
		pass

	# optional requests for samples
	def request_samples(self, sampler: 'spler.Sample', sample: 'spler.Sample', scene: 'scn.Scene'):
		pass














