'''
sample.py

pytracer.sampler package
The base class to model samples.

Created by Jiayao on July 31, 2017
Modified on Aug 13, 2017
'''
from __future__ import absolute_import
from pytracer import *

__all__ = ['CameraSample', 'Sample']


class CameraSample(object):
	'''
	CameraSample Class

	Also the baseclass for `Sample`
	'''
	def __init__(self, imageX: FLOAT=0., imageY: FLOAT=0.,
					lens_u: FLOAT=0., lens_v: FLOAT=0.,
					time: FLOAT=0.):
		self.imageX = imageX
		self.imageY = imageY
		self.lens_u = lens_u
		self.lens_v = lens_v
		self.time = time

	def __repr__(self):
		return "{}\nImage: ({}, {})\nLens:({}, {})\nTime: {}".format(self.__class__,
			self.imageX, self.imageY, self.lens_u, self.lens_v, self.time)


class Sample(CameraSample):
	def __init__(self, sampler: 'Sampler'=None, surf: 'SurfaceIntegrator'=None,
					vol: 'VolumeIntegrator'=None, scene: 'Scene'=None,
					imageX: FLOAT=0., imageY: FLOAT=0.,
					lens_u: FLOAT=0., lens_v: FLOAT=0.,
					time: FLOAT=0.):
		super().__init__(imageX, imageY, lens_u, lens_v, time)
		self.n1D = []
		self.n2D = []

		if surf is not None:
			surf.request_samples(sampler, self, scene)
		if vol is not None:
			vol.request_samples(sampler, self, scene)

		# init sample np arrays
		self.oneD = [None] * len(self.n1D)
		self.twoD = [None] * len(self.n2D)

	def duplicate(self, cnt: 'INT') -> ['Sample']:
		ret = [None] * cnt
		for i in range(cnt):
			smp = Sample()
			smp.n1D = self.n1D
			smp.n2D = self.n2D
			smp.oneD = [None] * len(self.n1D)
			smp.twoD = [None] * len(self.n2D)
			ret[i] = smp
		return ret

	def add_1d(self, num: INT) -> INT:
		'''
		Request `num` `Sample`s, index
		in the `n1D` is returned
		'''
		self.n1D.append(num)
		return len(self.n1D) - 1

	def add_2d(self, num: INT) -> INT:
		self.n2D.append(num)
		return len(self.n2D) - 1
