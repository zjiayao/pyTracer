"""
spectrum.py

The base class of the spectrum.

Created by Jiayao on July 30, 2017
"""
from __future__ import absolute_import
from typing import overload
from collections import (Iterable, Sequence, MutableSequence)
from abc import (ABCMeta, abstractmethod)
from enum import Enum
from pytracer import *
from pytracer.data.spectral import *

__all__ = ['xyz2rgb', 'rgb2xyz', 'Spectrum', 'SpectrumType', 'RGBSpectrum']


# Utility Declarations
SAMPLED_LAMBDA_START = 400
SAMPLED_LAMBDA_END = 700
N_SPECTRAL_SAMPLES = 30


def xyz2rgb(xyz: (Sequence, np.ndarray)) -> np.ndarray:
	return np.array([3.240479 * xyz[0] - 1.537150 * xyz[1] - 0.498535 * xyz[2],
	                 -0.969256 * xyz[0] + 1.875991 * xyz[1] + 0.041556 * xyz[2],
	                 0.055648 * xyz[0] - 0.204043 * xyz[1] + 1.057311 * xyz[2]])


def rgb2xyz(rgb: (Sequence, np.ndarray)) -> np.ndarray:
	return np.array([0.412453 * rgb[0] + 0.357580 * rgb[1] + 0.180423 * rgb[2],
	                 0.212671 * rgb[0] + 0.715160 * rgb[1] + 0.072169 * rgb[2],
	                 0.019334 * rgb[0] + 0.119193 * rgb[1] + 0.950227 * rgb[2]])


# Spectrum Definitions
class SpectrumType(Enum):
	REFLECTANCE = 0
	ILLUMINANT = 1


class CoefficientSpectrum(np.ndarray, metaclass=ABCMeta):
	"""
	CoefficientSpectrum Class

	Used to model the sample spectrum distribution
	and is to be subclassed by `RGBSpectrum` and
	`SampledSpectrum`
	"""
	def __new__(cls, n_samples: (INT, UINT), v: (FLOAT, float)=0.):
		return np.full(n_samples, v).view(cls)

	def __init__(self, n_samples: (INT, UINT), v: (FLOAT, float) = 0.):
		pass

	def __repr__(self):
		return "{}\nNumber of Samples: {}\nSamples: {}\n".format(self.__class__, self.n_samples, self.c)

	def __eq__(self, other):
		return np.array_equal(self, other)

	def __ne__(self, other):
		return not np.array_equal(self, other)

	@property
	def n_samples(self):
		return len(self)

	@property
	def c(self):
		return self

	@classmethod
	@overload
	def create(cls, sample: Iterable):
		pass

	@classmethod
	@overload
	def create(cls, spec: 'CoefficientSpectrum'):
		pass

	@classmethod
	def create(cls, arg):
		if isinstance(arg, CoefficientSpectrum):
			return arg.copy().view(cls)

		elif isinstance(arg, Iterable):
			if not np.ndim(arg) == 1:
				raise TypeError('{}-d np array cannot init {}'.format(np.ndim(arg), cls))

			if isinstance(arg, np.ndarray):
				return arg.copy().view(cls)

			return np.array(arg).view(cls)

		else:
			raise TypeError('Unsupported type {} for init {}'.format(type(arg), cls))
	#
	# def __add__(self, other: 'CoefficientSpectrum'):
	# 	return self.__class__.create(self.c + other.c)
	#
	# def __iadd__(self, other: 'CoefficientSpectrum'):
	# 	self.c += other.c
	# 	return self
	#
	# def __sub__(self, other: 'CoefficientSpectrum'):
	# 	return self.__class__.create(self.c - other.c)
	#
	# def __isub__(self, other):
	# 	self.c -= other.c
	# 	return self
	#
	# def __mul__(self, other):
	# 	if isinstance(other, CoefficientSpectrum):
	# 		return self.__class__.create(self.c * other.c)
	# 	else:
	# 		return self.__class__.create(self.c * other)
	#
	# def __imul__(self, other):
	# 	if isinstance(other, CoefficientSpectrum):
	# 		self.c *= other.c
	# 	else:
	# 		self.c *= other
	# 	return self
	#
	# def __rmul__(self, other):
	# 	if isinstance(other, CoefficientSpectrum):
	# 		return self.__class__.create(self.c * other.c)
	# 	else:
	# 		return self.__class__.create(self.c * other)
	#
	# def __div__(self, other):
	# 	if isinstance(other, CoefficientSpectrum):
	# 		return self.__class__.create(self.c / other.c)
	# 	else:
	# 		return self.__class__.create(self.c / other)
	#
	# def __truediv__(self, other):
	# 	if isinstance(other, CoefficientSpectrum):
	# 		return self.__class__.create(self.c / other.c)
	# 	else:
	# 		return self.__class__.create(self.c / other)
	#
	# def __itruediv__(self, other):
	# 	if isinstance(other, CoefficientSpectrum):
	# 		self.c /= other.c
	# 	else:
	# 		self.c /= other
	# 	return self
	#
	# def __idiv__(self, other):
	# 	if isinstance(other, CoefficientSpectrum):
	# 		self.c /= other.c
	# 	else:
	# 		self.c /= other
	# 	return self
	#
	# def __rtruediv__(self, other):
	# 	if isinstance(other, CoefficientSpectrum):
	# 		return self.__class__.create(other.c / other.c)
	# 	else:
	# 		return self.__class__.create(other / self.c)
	#
	# def __rdiv__(self, other):
	# 	if isinstance(other, CoefficientSpectrum):
	# 		return self.__class__.create(other.c / other.c)
	# 	else:
	# 		return self.__class__.create(other / self.c)
	#
	# def __neg__(self):
	# 	ret = self.__class__(self.n_samples)
	# 	ret.c = -self.c
	# 	return ret
	#
	# def __eq__(self, other: 'CoefficientSpectrum'):
	# 	return self.n_samples == other.n_samples and np.array_equal(self.c, other.c)
	#
	# def __ne__(self, other: 'CoefficientSpectrum'):
	# 	return not self == other

	def is_black(self) -> bool:
		"""
		Test whether the spectrum has
		zeros of SPD everywhere
		"""
		return (self < EPS).all()

	def sqrt(self) -> 'CoefficientSpectrum':
		"""
		Returns the square root
		of the current Spectrum
		"""
		return np.sqrt(self).view(self.__class__)

	def exp(self) -> 'CoefficientSpectrum':
		"""
		Returns the exponential
		of the current Spectrum
		"""
		return np.exp(self).view(self.__class__)

	def pow(self, n) -> 'CoefficientSpectrum':
		"""
		Returns the power
		of the current Spectrum
		"""
		return np.power(self, n).view(self.__class__)

	def lerp(self, t, other) -> 'CoefficientSpectrum':
		"""
		Returns the linear interpolation
		"""
		return util.ufunc_lerp(t, self.c, other.c).view(self.__class__)

	def has_nans(self) -> bool:
		return np.isnan(self).any()

	def clip(self, min=0., max=np.inf):
		return np.clip(self, min, max)

	@abstractmethod
	def y(self):
		raise NotImplementedError('src.core.spectrum.{}.y(): abstract method ' 
		                          'called'.format(self.__class__))

	@classmethod
	@abstractmethod
	def from_sampled(cls, lam: Iterable, v: Iterable):
		raise NotImplementedError('src.core.spectrum.{}.from_sampled(): abstract method ' 
		                          'called'.format(cls))

	@classmethod
	@abstractmethod
	def from_rgb(cls, rgb: Iterable, tp: 'SpectrumType' = SpectrumType.REFLECTANCE):
		raise NotImplementedError('src.core.spectrum.{}.from_rgb(): abstract method ' 
		                          'called'.format(cls))

	@classmethod
	@abstractmethod
	def from_xyz(cls, xyz: Iterable, tp: 'SpectrumType' = SpectrumType.REFLECTANCE):
		raise NotImplementedError('src.core.spectrum.{}.from_xyz(): abstract method ' 
		                          'called'.format(cls))

	@abstractmethod
	def to_xyz(self) -> 'SampledSpectrum':
		raise NotImplementedError('src.core.spectrum.{}.to_xyz(): abstract method ' 
		                          'called'.format(self.__class__))

	@abstractmethod
	def to_rgb(self) -> 'RGBSpectrum':
		raise NotImplementedError('src.core.spectrum.{}.to_rgb(): abstract method ' 
		                          'called'.format(self.__class__))


# TODO: Convertion with RGB is buggy
class SampledSpectrum(CoefficientSpectrum):
	"""
	SampledSpectrum Class

	Subclasses `CoefficientSpectrum`, used to
	model uniformly spaced samples taken from
	the spectrum.
	"""

	# XYZ SampledSpectrum
	X = None
	Y = None
	Z = None
	rgbRefl2SpectWhite = None
	rgbRefl2SpectCyan = None
	rgbRefl2SpectMagenta = None
	rgbRefl2SpectYellow = None
	rgbRefl2SpectRed = None
	rgbRefl2SpectGreen = None
	rgbRefl2SpectBlue = None
	rgbIllum2SpectWhite = None
	rgbIllum2SpectCyan = None
	rgbIllum2SpectMagenta = None
	rgbIllum2SpectYellow = None
	rgbIllum2SpectRed = None
	rgbIllum2SpectGreen = None
	rgbIllum2SpectBlue = None

	def __new__(cls, v: Iterable=None):
		if v is None:
			return np.full(N_SPECTRAL_SAMPLES, 0.).view(cls)
		elif isinstance(v, Iterable):
			return np.array(v).view(cls)
		raise TypeError('Unsupported type {} while initializing {}'.format(type(v), __class__))

	def __init__(self, v: Iterable=None):
		super().__init__(N_SPECTRAL_SAMPLES, v)
		pass

	def y(self):
		return np.sum(self.Y.dot(self)) * (SAMPLED_LAMBDA_END - SAMPLED_LAMBDA_START) \
										 / (CIE_Y_INTEGRAL * N_SPECTRAL_SAMPLES)

	@staticmethod
	def init():
		"""
		To be called at startup when pyTracer initialize
		"""
		SampledSpectrum.X = SampledSpectrum()
		SampledSpectrum.Y = SampledSpectrum()
		SampledSpectrum.Z = SampledSpectrum()
		SampledSpectrum.rgbRefl2SpectWhite = SampledSpectrum()
		SampledSpectrum.rgbRefl2SpectCyan = SampledSpectrum()
		SampledSpectrum.rgbRefl2SpectMagenta = SampledSpectrum()
		SampledSpectrum.rgbRefl2SpectYellow = SampledSpectrum()
		SampledSpectrum.rgbRefl2SpectRed = SampledSpectrum()
		SampledSpectrum.rgbRefl2SpectGreen = SampledSpectrum()
		SampledSpectrum.rgbRefl2SpectBlue = SampledSpectrum()
		SampledSpectrum.rgbIllum2SpectWhite = SampledSpectrum()
		SampledSpectrum.rgbIllum2SpectCyan = SampledSpectrum()
		SampledSpectrum.rgbIllum2SpectMagenta = SampledSpectrum()
		SampledSpectrum.rgbIllum2SpectYellow = SampledSpectrum()
		SampledSpectrum.rgbIllum2SpectRed = SampledSpectrum()
		SampledSpectrum.rgbIllum2SpectGreen = SampledSpectrum()
		SampledSpectrum.rgbIllum2SpectBlue = SampledSpectrum()

		wl0 = util.ufunc_lerp(np.arange(0, N_SPECTRAL_SAMPLES) / N_SPECTRAL_SAMPLES, SAMPLED_LAMBDA_START,
		                      SAMPLED_LAMBDA_END)
		wl1 = util.ufunc_lerp(np.arange(1, N_SPECTRAL_SAMPLES + 1) / N_SPECTRAL_SAMPLES, SAMPLED_LAMBDA_START,
		                      SAMPLED_LAMBDA_END)

		for i in range(N_SPECTRAL_SAMPLES):
			SampledSpectrum.X.c[i] = SampledSpectrum.average_spectrum_samples(CIE_LAMBDA, CIE_X, wl0[i],
			                                                                  wl1[i])
			SampledSpectrum.Y.c[i] = SampledSpectrum.average_spectrum_samples(CIE_LAMBDA, CIE_Y, wl0[i],
			                                                                  wl1[i])
			SampledSpectrum.Z.c[i] = SampledSpectrum.average_spectrum_samples(CIE_LAMBDA, CIE_Z, wl0[i],
			                                                                  wl1[i])
			SampledSpectrum.rgbRefl2SpectWhite.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                   RGBRefl2SpectWhite,
			                                                                                   wl0[i],
			                                                                                   wl1[i])
			SampledSpectrum.rgbRefl2SpectCyan.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                  RGBRefl2SpectCyan,
			                                                                                  wl0[i],
			                                                                                  wl1[i])
			SampledSpectrum.rgbRefl2SpectMagenta.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                     RGBRefl2SpectMagenta,
			                                                                                     wl0[i],
			                                                                                     wl1[i])
			SampledSpectrum.rgbRefl2SpectYellow.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                    RGBRefl2SpectYellow,
			                                                                                    wl0[i],
			                                                                                    wl1[i])
			SampledSpectrum.rgbRefl2SpectRed.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                 RGBRefl2SpectRed,
			                                                                                 wl0[i],
			                                                                                 wl1[i])
			SampledSpectrum.rgbRefl2SpectGreen.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                   RGBRefl2SpectGreen,
			                                                                                   wl0[i],
			                                                                                   wl1[i])
			SampledSpectrum.rgbRefl2SpectBlue.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                  RGBRefl2SpectBlue,
			                                                                                  wl0[i],
			                                                                                  wl1[i])
			SampledSpectrum.rgbIllum2SpectWhite.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                    RGBIllum2SpectWhite,
			                                                                                    wl0[i],
			                                                                                    wl1[i])
			SampledSpectrum.rgbIllum2SpectCyan.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                   RGBIllum2SpectCyan,
			                                                                                   wl0[i],
			                                                                                   wl1[i])
			SampledSpectrum.rgbIllum2SpectMagenta.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                      RGBIllum2SpectMagenta,
			                                                                                      wl0[i],
			                                                                                      wl1[i])
			SampledSpectrum.rgbIllum2SpectYellow.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                     RGBIllum2SpectYellow,
			                                                                                     wl0[i],
			                                                                                     wl1[i])
			SampledSpectrum.rgbIllum2SpectRed.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                  RGBIllum2SpectRed,
			                                                                                  wl0[i],
			                                                                                  wl1[i])
			SampledSpectrum.rgbIllum2SpectGreen.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                    RGBIllum2SpectGreen,
			                                                                                    wl0[i],
			                                                                                    wl1[i])
			SampledSpectrum.rgbIllum2SpectBlue.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda,
			                                                                                   RGBIllum2SpectBlue,
			                                                                                   wl0[i],
			                                                                                   wl1[i])

	@classmethod
	def from_sampled(cls, lam: Iterable, v: Iterable):
		# sort the spectrum if necessary
		v = np.array(v)[np.argsort(lam)]
		lam = np.sort(lam)

		# compute average SPD values 
		self = cls()
		for i in range(N_SPECTRAL_SAMPLES):
			l0 = util.lerp(i / N_SPECTRAL_SAMPLES, SAMPLED_LAMBDA_START, SAMPLED_LAMBDA_END)
			l1 = util.lerp(i+1 / N_SPECTRAL_SAMPLES, SAMPLED_LAMBDA_START, SAMPLED_LAMBDA_END)
			self[i] = cls.average_spectrum_samples(lam, v, l0, l1)

		return self

	@staticmethod
	def average_spectrum_samples(lam: MutableSequence, v: MutableSequence, ls: FLOAT, le: FLOAT) -> FLOAT:
		"""
		Piece-wise Linear Interpolation of
		SPD. 
		lam: Wavelength List
		v: Value List
		n: Number of Samples
		ls: Minimum Wavelength
		le: Maximum Wavelength

		NB: boundary values are used to extropolate
		out-of-bound values
		"""
		# out-of-bound or single sample
		if le <= lam[0]:
			return v[0]
		if ls >= lam[-1]:
			return v[-1]
		if len(lam) == 1:
			return v[0]

		v_sum = 0.
		# constant contributions before/after samples
		# if end points are partially covered
		if ls < lam[0]:
			v_sum += v[0] * (lam[0] - ls)
		if le > lam[-1]:
			v_sum += v[-1] * (le - lam[-1])

		# advance to the first and last contributable wavelength
		idx_lo = np.searchsorted(lam, ls) - 1
		idx_hi = np.searchsorted(lam, le) # due to np.searchsorted implementation

		if idx_lo >= 0:
			bk_lam_lo = lam[idx_lo]
			bk_v_lo = v[idx_lo]
			v[idx_lo] = util.lerp( (ls - bk_lam_lo) / (lam[idx_lo+1] - bk_lam_lo), v[idx_lo], v[idx_lo+1])
			lam[idx_lo] = ls			
		else:
			idx_lo = 0
			bk_lam_lo = lam[idx_lo]
			bk_v_lo = v[idx_lo]

		if idx_hi < len(lam) and lam[idx_hi] != le:
			bk_lam_hi = lam[idx_hi]
			bk_v_hi = v[idx_hi]
			v[idx_hi] = util.lerp((bk_lam_hi - le) / (lam[idx_hi] - lam[idx_hi-1]), v[idx_hi], v[idx_hi-1])
			lam[idx_hi] = le
			idx_hi += 1
		else:
			idx_hi = len(lam)
			bk_lam_hi = lam[-1]
			bk_v_hi = v[-1]

		# integrate using trapezoidal rule
		v_sum += np.trapz(v[idx_lo:idx_hi], lam[idx_lo:idx_hi])

		# clean-up
		lam[idx_lo] = bk_lam_lo
		lam[idx_hi-1] = bk_lam_hi
		v[idx_lo] = bk_v_lo
		v[idx_hi-1] = bk_v_hi

		return v_sum / (le - ls)

	@classmethod
	def from_rgb(cls, rgb: Sequence, tp: 'SpectrumType' = SpectrumType.REFLECTANCE):
		self = cls()
		lo, me, hi = np.argsort(rgb)

		if tp == SpectrumType.REFLECTANCE:
			# reflectance spectrum from RGB
			if lo == 0:
				spec = [cls.rgbRefl2SpectCyan, cls.rgbRefl2SpectGreen, cls.rgbRefl2SpectBlue]
			elif lo == 1:
				spec = [cls.rgbRefl2SpectMagenta, cls.rgbRefl2SpectRed, cls.rgbRefl2SpectBlue]
			else:
				spec = [cls.rgbRefl2SpectYellow, cls.rgbRefl2SpectRed, cls.rgbRefl2SpectGreen]

			self += rgb[lo] * cls.rgbRefl2SpectWhite
			self += (rgb[me] - rgb[lo]) * spec[lo]
			self += (rgb[hi] - rgb[me]) * spec[hi]

			self *= .94

		elif tp == SpectrumType.ILLUMINANT:
			# illuminance spectrum
			if lo == 0:
				spec = [cls.rgbIllum2SpectCyan, cls.rgbIllum2SpectGreen, cls.rgbIllum2SpectBlue]
			elif lo == 1:
				spec = [cls.rgbIllum2SpectMagenta, cls.rgbIllum2SpectRed, cls.rgbIllum2SpectBlue]
			else:
				spec = [cls.rgbIllum2SpectYellow, cls.rgbIllum2SpectRed, cls.rgbIllum2SpectGreen]

			self += rgb[lo] * cls.rgbIllum2SpectWhite
			self += (rgb[me] - rgb[lo]) * spec[lo]
			self += (rgb[hi] - rgb[me]) * spec[hi]

			self *= .86445

		else:
			raise TypeError

		return self.clip()

	@classmethod
	def from_xyz(cls, xyz: Sequence, tp: 'SpectrumType' = SpectrumType.REFLECTANCE):
		rgb = xyz2rgb(xyz)
		return cls.from_rgb(rgb, tp)

	def to_xyz(self) -> 'SampledSpectrum':
		"""Converts FULL SPECTRUM to xyz."""
		return (SampledSpectrum([SampledSpectrum.X.dot(self), SampledSpectrum.Y.dot(self), SampledSpectrum.Z.dot(self)]) *\
		       (SAMPLED_LAMBDA_END - SAMPLED_LAMBDA_START) / (CIE_Y_INTEGRAL * N_SPECTRAL_SAMPLES)).view(self.__class__)

	def to_rgb(self) -> 'RGBSpectrum':
		return RGBSpectrum.from_xyz(self.to_xyz())


class RGBSpectrum(CoefficientSpectrum):
	"""
	RGBSpectrum Class

	Subclasses `CoefficientSpectrum`, used to
	model RGB spectrum.
	"""
	def __new__(cls, v: Iterable=None):
		if v is None:
			return np.full(3, 0.).view(cls)
		elif isinstance(v, Iterable) and np.ndim(v) == 1 and np.shape(v)[0] == 3:
			return np.array(v).view(cls)
		raise TypeError('Unsupported type {} while initializing {}'.format(type(v), __class__))

	def __init__(self, v: Iterable=None):
		super().__init__(3)
		pass

	def init(self):
		pass

	@classmethod
	def create(cls, arg):
		if isinstance(arg, CoefficientSpectrum) and arg.n_samples == 3:
			return arg.copy().view(cls)

		elif isinstance(arg, Iterable):
			if np.ndim(arg) == 1 and np.shape(arg)[0] == 3:
				return np.ndarray(arg).view(cls)

		raise TypeError('Unsupported type {} for init {}'.format(type(arg), cls))

	def y(self) -> FLOAT:
		y_weight = np.array([0.212671, 0.715160, 0.072169])
		return y_weight.dot(self.c)

	@classmethod
	def from_sampled(cls, lam: Iterable, v: Iterable):
		return SampledSpectrum.from_sampled(lam, v).to_rgb()

	@classmethod
	def from_rgb(cls, rgb: Sequence, tp: 'SpectrumType'=SpectrumType.REFLECTANCE):
		return cls(rgb)

	@classmethod
	def from_xyz(cls, xyz: Sequence, tp: 'SpectrumType' = SpectrumType.REFLECTANCE):
		return cls.from_rgb(xyz2rgb(xyz), tp)

	def to_rgb(self) -> 'RGBSpectrum':
		return self.copy()

	def to_xyz(self) -> 'SampledSpectrum':
		return SampledSpectrum(rgb2xyz(self))


Spectrum = RGBSpectrum




