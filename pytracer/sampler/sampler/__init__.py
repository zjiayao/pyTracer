"""
__init__.py

pytracer.sampler.sampler package

Modelling samplers

Created by Jiayao on Aug 9, 2017
"""
from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
from pytracer import *


class Sampler(object, metaclass=ABCMeta):
	'''
	Sampler Class

	Generate a sequence of multidimensional sample
	positions, two for image dimensions, one for time
	and two more for lens positions.
	'''

	def __init__(self, xs: INT, xe: INT, ys: INT, ye: INT,
	             spp: INT, s_open: FLOAT, s_close: FLOAT):
		'''
		Range for samples:
		[xs, xe-1], [ys, ye-1]
		'''
		self.xPixel_start = xs
		self.xPixel_end = xe
		self.yPixel_strat = ys
		self.yPixel_end = ye
		self.spp = spp  # samples per pixel
		self.s_open = s_open
		self.s_close = s_close
		self.has_more_samples = False

	def __repr__(self):
		return "{}\nx: {} - {}\ny: {} - {}\nSamples per pixel: {}" \
			.format(self.__class__, self.xPixel_start, self.xPixel_end,
		            self.yPixel_strat, self.yPixel_end, self.spp)

	def __iter__(self):
		return self

	@abstractmethod
	def __next__(self, rng=None) -> 'np.ndarray':
		'''
		analogous to get_more_samples() in pbrt

		rng is the seed.
		returns an np object array holding samples,
		zero if all samples has been generated
		'''
		raise NotImplementedError('src.core.sampler {}.__next__(): abstract method called' \
		                          .format(self.__class__))

	@abstractmethod
	def maximum_sample_cnt(self) -> INT:
		raise NotImplementedError('src.core.sampler {}.maximum_sample_cnt(): abstract method '
		                          'called'.format(self.__class__))

	@abstractmethod
	def get_subsampler(self, num: INT, cnt: INT) -> 'Sampler':
		'''
		Get the subsamplers used.
		num ranges from 0 to cnt-1

		Generally decompose the image into
		rectangular windows and use one subsampler
		per each.
		'''
		raise NotImplementedError('src.core.sampler {}.get_subsampler(): abstract method '
		                          'called'.format(self.__class__))

	def report_results(self, samples: ['Sample'], rays: ['RayDifferential'],
	                   ls: 'Spectrum', isects: ['Intersection']) -> bool:
		'''
		For adaptive samplers
		'''
		return True

	@abstractmethod
	def round_size(self, size: INT) -> INT:
		'''
		round_size

		Return a convenient size of sampler
		as requested by an `Integrator`
		'''
		raise NotImplementedError('src.core.sampler {}.round_size(): abstract method '
		                          'called'.format(self.__class__))

	def compute_subwindow(self, num: INT, cnt: INT) -> 'np.ndarray':
		'''
		Computes a pixel sampling range
		given a tile number and a total number
		of tiles count.
		Returns:
		[xs, xe, ys, ye]
		'''
		# determine the number of tiles in each dimension
		# in effort to make square tiles
		dx = self.xPixel_end - self.xPixel_start
		dy = self.yPixel_end - self.yPixel_strat
		nx = cnt
		ny = 1
		while nx & 0x1 == 0 and 2 * dx * ny < dy * nx:
			nx >>= 1
			ny <<= 1

		# compute x and y pixel sample range
		x0 = num % nx
		y0 = num // nx
		x = np.clip([x0 / nx, (x0 + 1) / nx], self.xPixel_start, self.xPixel_end)
		y = np.clip([y0 / ny, (y0 + 1) / ny], self.yPixel_strat, self.yPixel_end)
		return np.concatenate([util.ufunc_lerp(x, self.xPixel_start, self.xPixel_end),
		                       util.ufunc_lerp(y, self.yPixel_strat, self.yPixel_end)])


from pytracer.sampler.sampler.stratified import *