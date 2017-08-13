"""
filter.py

The base class to model filters.

Created by Jiayao on Aug 1, 2017
"""
from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
from pytracer import *

__all__ = ['Filter', 'BoxFilter', 'TriangleFilter',
           'GaussianFilter', 'MitchellFilter', 'LanczosSincFilter']


class Filter(object, metaclass=ABCMeta):
	"""
	Filter class
	"""

	def __init__(self, xw: FLOAT, yw: FLOAT):
		self.xw = xw
		self.yw = yw
		self.xwInv = 1. / xw
		self.ywInv = 1. / yw

	def __repr__(self):
		return "{}\n{} * {}".format(self.__class__, self.xw, self.yw)

	@abstractmethod
	def __call__(self, x: FLOAT, y: FLOAT) -> FLOAT:
		raise NotImplementedError('src.core.filter {}.__call__(): abstract method '
									'called'.format(self.__class__)) 		


class BoxFilter(Filter):
	def __call__(self, x: FLOAT, y: FLOAT) -> FLOAT:
		return 1.


class TriangleFilter(Filter):
	def __call__(self, x: FLOAT, y: FLOAT) -> FLOAT:
		return np.fmax(0., self.xw - np.fabs(x)) * np.fmax(0., self.yw - np.fabs(y))


class GaussianFilter(Filter):
	def __init__(self, x: FLOAT, y: FLOAT, alpha: FLOAT):
		super().__init__(x, y)
		self.alpha = alpha
		self.expX = np.exp(-alpha * x * x)
		self.expY = np.exp(-alpha * y * y)

	def __call__(self, x: FLOAT, y: FLOAT) -> FLOAT:
		return self.gaussian(x, self.expX) * self.gaussian(y, self.expY)

	def gaussian(self, d: FLOAT, expv: FLOAT) -> FLOAT:
		return np.fmax(0., np.exp(-self.alpha * d * d) - expv)


class MitchellFilter(Filter):
	def __init__(self, b: FLOAT, c:FLOAT, xw: FLOAT, yw: FLOAT):
		super().__init__(xw, yw)
		self.B = b
		self.C = c

	def __call__(self, x: FLOAT, y: FLOAT) -> FLOAT:
		return self.mitchell(np.multiply(x, self.xwInv)) * self.mitchell(np.multiply(y, self.ywInv))

	def mitchell(self, x: FLOAT) -> FLOAT:
		x = np.fabs(2. * x)
		# add np.ndarray support
		if hasattr(x, '__iter__'):
			for i, xx in enumerate(x):
				if xx > 1.:
					x[i] = ((-self.B - 6 * self.C) * xx * xx * xx + (6 * self.B + 30 * self.C) * xx * xx +
						(-12 * self.B - 48 * self.C) * xx + (8 * self.B + 24 * self.C)) * (1./6.)
				else:
					x[i] = ((12 - 9 * self.B - 6 * self.C) * xx * xx * xx +
					(-18 + 12 * self. B + 6 * self.C) * xx * xx +
					(6 - 2 * self.B)) * (1./6.);
			return np.array(x)

		if x > 1.:
			return ((-self.B - 6 * self.C) * x * x * x + (6 * self.B + 30 * self.C) * x * x +
						(-12 * self.B - 48 * self.C) * x + (8 * self.B + 24 * self.C)) * (1./6.)
		else:
			return  ((12 - 9 * self.B - 6 * self.C) * x * x * x +
					(-18 + 12 * self. B + 6 * self.C) * x * x +
					(6 - 2 * self.B)) * (1./6.);


class LanczosSincFilter(Filter):
	def __init__(self, xw: FLOAT, yw: FLOAT, tau: FLOAT):
		super().__init__(xw, yw)
		self.tau = tau

	def __call__(self, x: FLOAT, y: FLOAT) -> FLOAT:
		return self.sinc(np.multiply(x, self.xwInv)) * self.sinc(np.multiply(y, self.ywInv))

	def sinc(self, x: FLOAT) -> FLOAT:
		x = np.fabs(x)

		# add np.ndarray or list support
		if hasattr(x, '__iter__'):
			for i, xx in enumerate(x):
				if xx < EPS:
					x[i] = 1.
				elif xx > 1.:
					x[i] = 0.
				else:
					x[i] = np.sinc(xx * self.tau) * np.sinc(xx)
			return np.array(x)

		if x < EPS:
			return 1.
		elif x > 1.:
			return 0.
		return np.sinc(x * self.tau) * np.sinc(x)

