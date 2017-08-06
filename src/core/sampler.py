'''
sampler.py

The base class to model samplers.

Created by Jiayao on July 31, 2017
'''

from numba import jit
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
from src.core.pytracer import *


@jit
def stratified_sample_1d(nSamples: INT, jitter: bool=True, rng=np.random.rand) -> 'np.ndarray':
	if not jitter:
		return (np.arange(nSamples) + .5) / nSamples
	else:
		return (np.arange(nSamples) + rng(nSamples)) / nSamples

@jit
def stratified_sample_2d(nx: INT, ny: INT, jitter: bool=True, rng=np.random.rand) -> 'np.ndarray':

	if jitter:
		return np.column_stack([(np.tile(np.arange(nx),   ny) + rng(nx * ny)) / nx,
								(np.repeat(np.arange(ny), nx) + rng(nx * ny)) / ny])
	else:
		return np.column_stack([(np.tile(np.arange(nx),   ny) + .5) / nx,
								(np.repeat(np.arange(ny), nx) + .5) / ny])

@jit
def latin_hypercube_1d(nSamples: INT, rng=np.random.rand) -> 'np.ndarray':
	ret = (np.arange(nSamples) + rng(nSamples)) / nSamples
	np.random.shuffle(ret)
	return ret

@jit
def latin_hypercube_2d(n: INT, rng=np.random.rand) -> 'np.ndarray':
	ys = (np.arange(n) + rng(n)) / n
	np.random.shuffle(ys)
	return np.column_stack([(np.arange(n) + rng(n)) / n, ys])

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
		self.oneD = [None for _ in range(len(self.n1D))]
		self.twoD = [None for _ in range(len(self.n2D))]

	def duplicate(self, cnt: 'INT') -> ['Sample']:
		ret = []
		for i in range(cnt):
			smp = Sample()
			smp.n1D = self.n1D.copy()
			smp.n2D = self.n2D.copy()
			smp.oneD = [None for _ in range(len(smp.n1D))]
			smp.twoD = [None for _ in range(len(smp.n2D))]
			ret.append(smp)
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
		self.spp = spp # samples per pixel
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

	@jit
	def compute_subwindow(self, num:INT, cnt: INT) -> 'np.ndarray':
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
		x = np.clip([x0/nx, (x0+1)/nx], self.xPixel_start, self.xPixel_end)
		y = np.clip([y0/ny, (y0+1)/ny], self.yPixel_strat, self.yPixel_end)
		return np.concatenate([ufunc_lerp(x, self.xPixel_start, self.xPixel_end),
							   ufunc_lerp(y, self.yPixel_strat, self.yPixel_end)])


class StratifiedSampler(Sampler):
	'''
	StratifiedSampler Class

	Subclasses `Sampler`. Jittering
	stratified sampling is implemented.
	Traversal pattern;
	---------/
	---------/
	--------->
	'''
	def __init__(self, xs: INT, xe: INT, ys: INT, ye: INT,
					xst: INT, yst: INT, jitter: bool, s_open: FLOAT, s_close: FLOAT):
		super().__init__(xs, xe, ys, ye, xst * yst, s_open, s_close)
		self.jitter = jitter
		self.xPos = xs # current position
		self.yPos = ys
		self.xSamples = xst
		self.ySamples = yst

	@jit
	def __next__(self, rng=np.random.rand) -> 'np.ndarray':
		'''
		returns an np object array holding `Sample`s
		'''
		if self.yPos == self.yPixel_end:
			raise StopIteration
			return None
		nSamples = self.xSamples * self.ySamples
		# generate stratified samples
		imageSamples = stratified_sample_2d(self.xSamples, self.ySamples, self.jitter, rng)
		lensSamples = stratified_sample_2d(self.xSamples, self.ySamples, self.jitter, rng)
		timeSamples = stratified_sample_1d(self.xSamples * self.ySamples, self.jitter, rng)

		## shift samples to pixel coord
		imageSamples += [self.xPos, self.yPos]

		## decorrelate dimensions
		np.random.shuffle(lensSamples)
		np.random.shuffle(timeSamples)

		## init `Sample` object
		samples = np.empty(nSamples, dtype=object)
		for i in range(nSamples):
			samples[i] = Sample(imageX=imageSamples[i, 0], imageY=imageSamples[i, 1],
				lens_u=lensSamples[i, 0], lens_v=lensSamples[i, 1],
				time=Lerp(timeSamples[i], self.s_open, self.s_close))

		## generate patterns for integraters, if needed
			for j, n in enumerate(samples[i].n1D):
				samples[i].oneD[j] = latin_hypercube_1d(n, rng)
			for j, n in enumerate(samples[i].n2D):
				samples[i].twoD[j] = latin_hypercube_2d(n, rng)


		# advance current position
		self.xPos += 1
		if self.xPos == self.xPixel_end:
			self.xPos = self.xPixel_start
			self.yPos += 1

		return samples

	def round_size(self, size: INT) -> INT:
		'''
		round_size

		`StratifiedSampler` has no preferences
		'''
		return size

	def maximum_sample_cnt(self) -> INT:
		return self.xSamples * self.ySamples


	def get_subsampler(self, num: INT, cnt: INT) -> 'Sampler':
		'''
		Returns `None` if operation
		cannot be done
		'''
		ret = self.compute_subwindow(num, cnt)
		if ret[0] == ret[1] or ret[2] == ret[3]:
			return None
		return StratifiedSampler(ret[0], ret[1], ret[2], ret[3],
					self.xSamples, self.ySamples, self.jitter, self.s_open, self.s_close)

class AdaptiveTest(Enum):
	ADAPTIVE_COMPARE_SHAPE_ID = 0
	ADAPTIVE_CONTRAST_THRESHOLD = 1

class StratifiedSampler(Sampler):
	'''
	StratifiedSampler Class

	Subclasses `Sampler`. Jittering
	stratified sampling is implemented.
	Traversal pattern;
	---------/
	---------/
	--------->
	'''
	def __init__(self, xs: INT, xe: INT, ys: INT, ye: INT,
					xst: INT, yst: INT, supersamplePixel: bool, s_open: FLOAT, s_close: FLOAT):
		super().__init__(xs, xe, ys, ye, xst * yst, s_open, s_close)
		self.jitter = jitter
		self.xPos = xs # current position
		self.yPos = ys
		self.xSamples = xst
		self.ySamples = yst

	@jit
	def __next__(self, rng=np.random.rand) -> 'np.ndarray':
		'''
		returns an np object array holding `Sample`s
		'''
		if self.yPos == self.yPixel_end:
			raise StopIteration
			return None
		nSamples = self.xSamples * self.ySamples
		# generate stratified samples
		imageSamples = stratified_sample_2d(self.xSamples, self.ySamples, self.jitter, rng)
		lensSamples = stratified_sample_2d(self.xSamples, self.ySamples, self.jitter, rng)
		timeSamples = stratified_sample_1d(self.xSamples * self.ySamples, self.jitter, rng)

		## shift samples to pixel coord
		imageSamples += [self.xPos, self.yPos]

		## decorrelate dimensions
		np.random.shuffle(lensSamples)
		np.random.shuffle(timeSamples)

		## init `Sample` object
		samples = np.empty(nSamples, dtype=object)
		for i in range(nSamples):
			samples[i] = Sample(imageX=imageSamples[i, 0], imageY=imageSamples[i, 1],
				lens_u=lensSamples[i, 0], lens_v=lensSamples[i, 1],
				time=Lerp(timeSamples[i], self.s_open, self.s_close))

		## generate patterns for integraters, if needed
			for j, n in enumerate(samples[i].n1D):
				samples[i].oneD[j] = latin_hypercube_1d(n, rng)
			for j, n in enumerate(samples[i].n2D):
				samples[i].twoD[j] = latin_hypercube_2d(n, rng)


		# advance current position
		self.xPos += 1
		if self.xPos == self.xPixel_end:
			self.xPos = self.xPixel_start
			self.yPos += 1

		return samples

	def round_size(self, size: INT) -> INT:
		'''
		round_size

		`StratifiedSampler` has no preferences
		'''
		return size

	def maximum_sample_cnt(self) -> INT:
		return self.xSamples * self.ySamples


	def get_subsampler(self, num: INT, cnt: INT) -> 'Sampler':
		'''
		Returns `None` if operation
		cannot be done
		'''
		ret = self.compute_subwindow(num, cnt)
		if ret[0] == ret[1] or ret[2] == ret[3]:
			return None
		return StratifiedSampler(ret[0], ret[1], ret[2], ret[3],
					self.xSamples, self.ySamples, self.jitter, self.s_open, self.s_close)
