'''
spectrum.py

The base class of the spectrum.

Created by Jiayao on July 27, 2017
'''

import numpy as np
from src.core.pytracer import *

# Utility Declarations

SAMPLED_LAMBDA_START = 400
SAMPLED_LAMBDA_END = 700
N_SPECTRAL_SAMPLES = 30

class CoefficientSpectrum(object):
	'''
	CoefficientSpectrum Class

	Used to model the sample spectrum distribution
	and is to be subclassed by `RGBSpectrum` and
	`SampleSpectrum`
	'''
	def __init__(self, nSamples, v: FLOAT = 0.):
		self.nSamples = nSamples
		self.c = np.full(nSamples, v)

	def __repr__(self):
		return "{}\nNumber of Samples: {}".format(self.__class__, self.nSamples)

	@classmethod
	def fromArray(cls, sample: 'np.ndarray'):
		if sample is None:
			raise ValueError('Cannot initialized {} from None'.format(cls))		
		self = cls(len(sample))
		self.c = sample.copy()
		return self

	@classmethod
	def fromSpectrum(cls, spec: 'CoefficientSpectrum'):
		if sample is None:
			raise ValueError('Cannot initialized {} from None'.format(cls))		
		self.nSamples = cls(spec.nSamples)
		self.c = spec.c.copy()
		return self

	def __add__(self, other):
		return CoefficientSpectrum.fromArray(self.c + other.c)

	def __sub__(self, other):
		return CoefficientSpectrum.fromArray(self.c - other.c)

	def __mul__(self, other):
		return CoefficientSpectrum.fromArray(self.c * other.c)

	def __div__(self, other):
		return CoefficientSpectrum.fromArray(self.c / other.c)

	def __neg__(self):
		ret = CoefficientSpectrum(self.nSamples)
		ret.c = -self.c
		return ret

	def __eq__(self, other):
		return np.array_equal(self.c, other.c)

	def __ne__(self, other):
		return not np.array_equal(self.c, other.c)

	def is_black(self) -> bool:
		'''
		Test whether the spectrum has
		zeros of SPD everywhere
		'''
		return (self.c == 0.).all()

	def sqrt(self) -> 'CoefficientSpectrum':
		'''
		Returns the square root
		of the current Spectrum
		'''
		ret = CoefficientSpectrum(self.nSamples)
		ret.c = np.sqrt(c)
		return ret

	def exp(self) -> 'CoefficientSpectrum':
		'''
		Returns the exponential
		of the current Spectrum
		'''
		ret = CoefficientSpectrum(self.nSamples)
		ret.c = np.exp(c)
		return ret

	def pow(self, n) -> 'CoefficientSpectrum':
		'''
		Returns the power
		of the current Spectrum
		'''
		ret = CoefficientSpectrum(self.nSamples)
		ret.c = np.power(ret.c, n)
		return ret

	def lerp(self, t, other) -> 'CoefficientSpectrum':
		'''
		Returns the linear intropolation
		'''
		ret = CoefficientSpectrum(self.nSamples)
		ret.c = ufunc_lerp(t, self.c, other.c)
		return ret

	def clip(self, low: FLOAT = 0., high: FLOAT = np.inf):
		ret =CoefficientSpectrum(self.nSamples)
		ret.c = np.clip(self.c, low, high)
		return ret

	def has_nans(self) -> bool:
		return np.isnan(self.c).any()

class SampledSpectrum(CoefficientSpectrum):
	'''
	SampledSpectrum Class

	Subclasses `CoefficientSpectrum`, used to
	model uniformly spaced samples taken from
	the spectrum.
	'''
	def __init__(self, v: FLOAT = 0.):
		super().__init__(N_SPECTRAL_SAMPLES, v)

	@classmethod
	def fromSampled(cls, lam: [FLOAT], v: [FLOAT], n: INT):
		# sort the spectrum if necessary

		v = v[np.argsort(lam)]
		lam = np.sort(lam)

		# compute average SPD values 
		r = SampledSpectrum()
		for i in range(N_SPECTRAL_SAMPLES):
			l0 = Lerp(i / N_SPECTRAL_SAMPLES, SAMPLED_LAMBDA_START, SAMPLED_LAMBDA_END)
			l1 = Lerp(i+1 / N_SPECTRAL_SAMPLES, SAMPLED_LAMBDA_START, SAMPLED_LAMBDA_END)
			r.c[i] = SampleSpectrum.average_spectrum_samples(lam, v, n, l0, l1)

		return r

	@staticmethod
	def average_spectrum_samples(lam: [FLOAT], v: [FLOAT], n: INT, ls: FLOAT, le: FLOAT):
		'''
		Piece-wise Linear Interpolation of
		SPD. 
		lam: Wavelength List
		v: Value List
		n: Number of Samples
		ls: Minimum Wavelength
		le: Maximum Wavelength

		NB: boundary values are used to extropolate
		out-of-bound values
		'''
		# out-of-bound or single sample
		if le <= lam[0]:
			return v[0]
		if ls >= lam[-1]:
			return v[-1]
		if n == 1:
			return v[0]

		sum = 0.
		# constant contributions before/after samples
		# if end points are partially covered
		if ls < lam[0]:
			sum += v[0] * (lam[0] - ls)
		if le > lam[-1]:
			sum += v[-1] * (le - lam[-1])

		# advance to the first and last contributable wavelength
		idx_lo = np.searchsorted(lam, ls) - 1
		idx_hi = np.searchsorted(lam, le) # due to np.searchsorted implementation

		if idx_lo >= 0:
			bk_lam_lo = lam[idx_lo]
			bk_v_lo = v[idx_lo]
			lam[idx_lo] = ls
			v[idx_lo] = Lerp( (ls - bk_lam_lo) / (lam[idx_lo+1] - bk_lam_lo), v[idx_lo], v[idx_lo+1])
		else:
			idx_lo = 0
			bk_lam_lo = lam[idx_lo]
			bk_v_lo = v[idx_lo]

		if idx_hi < len(lam):
			bk_lam_hi = lam[idx_hi]
			bk_v_hi = v[idx_hi]
			lam[idx_hi] = le
			v[idx_hi] = Lerp( (le - lam[idx_hi-1]) / (lam[idx_hi] - lam[idx_hi-1]), v[idx_hi], v[idx_hi-1])
			idx_hi += 1
		else:
			idx_hi = len(lam)
			bk_lam_hi = lam[-1]
			bk_v_hi = v[-1]


		# integrate using trapezoidal rule
		sum += ny.trapz(v[idx_lo:idx_hi], lam[idx_lo:idx_hi])

		# clean-up
		lam[idx_lo] = bk_lam_lo
		lam[idx_hi-1] = bk_
		v[idx_lo] = bk_v_lo
		v[idx_hi-1] = bk_v_hi

















