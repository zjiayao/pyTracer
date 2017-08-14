"""
integrator.py

Model integrators.

Created by Jiayao on Aug 9, 2017
"""

from __future__ import absolute_import
from abc import ABCMeta

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
	def preprocess(self, scene: 'Scene', camera: 'Camera', renderer:'Renderer'):
		pass

	# optional requests for samples
	def request_samples(self, sampler: 'Sample', sample: 'Sample', scene: 'Scene'):
		pass














