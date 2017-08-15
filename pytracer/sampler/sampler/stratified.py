"""
stratified.py


pytracer.sampler.sampler package

Using stratified sampling.

Created by Jiayao on July 31, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer import *
from pytracer.sampler.sample import Sample
from pytracer.sampler.sampler import Sampler
from pytracer.sampler.utility import (stratified_sample_1d, stratified_sample_2d,
                                      latin_hypercube_1d, latin_hypercube_2d)

__all__ = ['StratifiedSampler']


class StratifiedSampler(Sampler):
	"""
	StratifiedSampler Class

	Subclasses `Sampler`. Jittering
	stratified sampling is implemented.
	Traversal pattern;
	---------/
	---------/
	--------->
	"""

	def __init__(self, xs: INT, xe: INT, ys: INT, ye: INT,
	             xst: INT, yst: INT, jitter: bool, s_open: FLOAT, s_close: FLOAT):
		xst = max(xst, 1)
		yst = max(yst, 1)
		super().__init__(xs, xe, ys, ye, xst * yst, s_open, s_close)
		self.jitter = jitter
		self.xPos = xs  # current position
		self.yPos = ys
		self.xSamples = xst
		self.ySamples = yst

	def __next__(self, rng=np.random.rand) -> 'np.ndarray':
		"""
		returns an np object array holding `Sample`s
		"""
		if self.yPos == self.yPixel_end:
			raise StopIteration

		n_samples = self.xSamples * self.ySamples
		# generate stratified samples
		image_samples = stratified_sample_2d(self.xSamples, self.ySamples, self.jitter, rng)
		lens_samples = stratified_sample_2d(self.xSamples, self.ySamples, self.jitter, rng)
		time_samples = stratified_sample_1d(self.xSamples * self.ySamples, self.jitter, rng)

		# shift samples to pixel coord
		image_samples += [self.xPos, self.yPos]

		# decorrelate dimensions
		np.random.shuffle(lens_samples)
		np.random.shuffle(time_samples)

		# init `Sample` object
		samples = np.empty(n_samples, dtype=object)
		for i in range(n_samples):
			samples[i] = Sample(imageX=image_samples[i, 0], imageY=image_samples[i, 1],
			                    lens_u=lens_samples[i, 0], lens_v=lens_samples[i, 1],
			                    time=util.lerp(time_samples[i], self.s_open, self.s_close))

			# generate patterns for integraters, if needed
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
		"""
		round_size

		`StratifiedSampler` has no preferences
		"""
		return size

	def maximum_sample_cnt(self) -> INT:
		return self.xSamples * self.ySamples

	def get_subsampler(self, num: INT, cnt: INT) -> 'Sampler':
		"""
		Returns `None` if operation
		cannot be done
		"""
		ret = self.compute_subwindow(num, cnt)
		if ret[0] == ret[1] or ret[2] == ret[3]:
			return None
		return StratifiedSampler(ret[0], ret[1], ret[2], ret[3],
		                         self.xSamples, self.ySamples, self.jitter, self.s_open, self.s_close)