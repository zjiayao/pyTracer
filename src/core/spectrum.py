'''
spectrum.py

The base class of the spectrum.

Created by Jiayao on July 30, 2017
'''
from numba import jit
import numpy as np
from enum import Enum
from src.core.pytracer import *
# Utility Declarations

SAMPLED_LAMBDA_START = 400
SAMPLED_LAMBDA_END = 700
N_SPECTRAL_SAMPLES = 30

@jit
def xyz2rgb(xyz: [FLOAT]) -> [FLOAT]:
	return np.array([ 3.240479 * xyz[0] - 1.537150 * xyz[1] - 0.498535 * xyz[2],
					-0.969256 * xyz[0] + 1.875991 * xyz[1] + 0.041556 * xyz[2],
					 0.055648 * xyz[0] - 0.204043 * xyz[1] + 1.057311 * xyz[2]])

@jit
def rgb2xyz(rgb: [FLOAT]) -> [FLOAT]:
	return np.array([0.412453 * rgb[0] + 0.357580 * rgb[1] + 0.180423 * rgb[2],
					0.212671 * rgb[0] + 0.715160 * rgb[1] + 0.072169 * rgb[2],
					0.019334 * rgb[0] + 0.119193 * rgb[1] + 0.950227 * rgb[2]])

# Spectral Data
from src.spectral.data import *

# Spectrum Definitions
class SpectrumType(Enum):
	REFLECTANCE = 0
	ILLUMINANT = 1


class CoefficientSpectrum(object):
	'''
	CoefficientSpectrum Class

	Used to model the sample spectrum distribution
	and is to be subclassed by `RGBSpectrum` and
	`SampledSpectrum`
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

	# XYZ SampledSpectrum
	X = None
	Y = None
	Z = None
	rgbRefl2SpectWhite  = None
	rgbRefl2SpectCyan  = None
	rgbRefl2SpectMagenta  = None
	rgbRefl2SpectYellow  = None
	rgbRefl2SpectRed  = None
	rgbRefl2SpectGreen  = None
	rgbRefl2SpectBlue  = None
	rgbIllum2SpectWhite  = None
	rgbIllum2SpectCyan  = None
	rgbIllum2SpectMagenta  = None
	rgbIllum2SpectYellow  = None
	rgbIllum2SpectRed  = None
	rgbIllum2SpectGreen  = None
	rgbIllum2SpectBlue  = None

	def __init__(self, v: FLOAT = 0.):
		super().__init__(N_SPECTRAL_SAMPLES, v)

	@classmethod
	@jit
	def fromSampled(cls, lam: [FLOAT], v: [FLOAT], n: INT):
		# sort the spectrum if necessary

		v = v[np.argsort(lam)]
		lam = np.sort(lam)

		# compute average SPD values 
		self = cls()
		for i in range(N_SPECTRAL_SAMPLES):
			l0 = Lerp(i / N_SPECTRAL_SAMPLES, SAMPLED_LAMBDA_START, SAMPLED_LAMBDA_END)
			l1 = Lerp(i+1 / N_SPECTRAL_SAMPLES, SAMPLED_LAMBDA_START, SAMPLED_LAMBDA_END)
			self.c[i] = cls.average_spectrum_samples(lam, v, n, l0, l1)

		return self

	@staticmethod
	@jit
	def average_spectrum_samples(lam: [FLOAT], v: [FLOAT], n: INT, ls: FLOAT, le: FLOAT) -> FLOAT:
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

		#print("idx_lo: {}".format(idx_lo))
		#print("idx_hi: %d" % idx_hi)
		if idx_lo >= 0:
			bk_lam_lo = lam[idx_lo]
			bk_v_lo = v[idx_lo]
			v[idx_lo] = Lerp( (ls - bk_lam_lo) / (lam[idx_lo+1] - bk_lam_lo), v[idx_lo], v[idx_lo+1])
			lam[idx_lo] = ls			
		else:
			idx_lo = 0
			bk_lam_lo = lam[idx_lo]
			bk_v_lo = v[idx_lo]

		if idx_hi < len(lam) and lam[idx_hi] != le:
			bk_lam_hi = lam[idx_hi]
			bk_v_hi = v[idx_hi]
			v[idx_hi] = Lerp( (bk_lam_hi - le) / (lam[idx_hi] - lam[idx_hi-1]), v[idx_hi], v[idx_hi-1])
			lam[idx_hi] = le
			idx_hi += 1
		else:
			idx_hi = len(lam)
			bk_lam_hi = lam[-1]
			bk_v_hi = v[-1]

		#print(lam)
		#print(v)


		# integrate using trapezoidal rule
		sum += np.trapz(v[idx_lo:idx_hi], lam[idx_lo:idx_hi])

		# clean-up
		lam[idx_lo] = bk_lam_lo
		lam[idx_hi-1] = bk_lam_hi
		v[idx_lo] = bk_v_lo
		v[idx_hi-1] = bk_v_hi

		#print(lam)
		#print(v)

		return sum / (le - ls)

	@classmethod
	@jit
	def fromRGB(cls, rgb: [FLOAT], tp: 'SpectrumType'):
		self = cls()
		hi, me, lo = np.argsort(np.array(rgb))

		if tp == SpectrumType.REFLECTANCE:
			# relfectance spectrum from RGB
			sepc = [cls.rgbRefl2SpectCyan,
					cls.rgbRefl2SpectBlue,
					cls.rgbRefl2SpectGreen]

			self += rgb[lo] * cls.rgbRefl2SpectWhite
			self += (rgb[me] - rgb[lo]) * spec[lo]
			self += (rgb[hi] - rgb[me]) * sepc[hi]

		elif tp == SpectrumType.ILLUMINANT:
			# illuminance spectrum
			spec = [cls.rgbIllum2SpectCyan,
					cls.rgbIllum2SpectBlue,
					cls.rgbIllum2SpectGreen]

			self += rgb[lo] * cls.rgbIllum2SpectWhite
			self += (rgb[me] - rgb[lo]) * spec[lo]
			self += (rgb[hi] - rgb[me]) * sepc[hi]

		else:
			raise TypeError

		return self.clip()

	@classmethod
	@jit
	def fromXYZ(cls, xyz: [FLOAT], tp: 'SpectrumType' = SpectrumType.REFLECTANCE):
		rgb = xyz2rgb(xyz)
		return cls.fromRGB(rgb)

	@classmethod
	@jit
	def fromRGBSpectrum(cls, r: 'RGBSpectrum', t: 'SpectrumType'):
		rgb = r.toRGB()
		return cls.fromRGB(rbg, t)

	@staticmethod
	def init():
		'''
		To be called at startup when pyTracer initialize
		'''
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

		wl0 = ufunc_lerp(np.arange(0, N_SPECTRAL_SAMPLES)/N_SPECTRAL_SAMPLES, SAMPLED_LAMBDA_START, SAMPLED_LAMBDA_END)
		wl1 = ufunc_lerp(np.arange(1, N_SPECTRAL_SAMPLES+1)/N_SPECTRAL_SAMPLES, SAMPLED_LAMBDA_START, SAMPLED_LAMBDA_END)
		
		for i in range(N_SPECTRAL_SAMPLES):
			SampledSpectrum.X.c[i] = SampledSpectrum.average_spectrum_samples(CIE_LAMBDA, CIE_X, N_CIE_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.Y.c[i] = SampledSpectrum.average_spectrum_samples(CIE_LAMBDA, CIE_Y, N_CIE_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.Z.c[i] = SampledSpectrum.average_spectrum_samples(CIE_LAMBDA, CIE_Z, N_CIE_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbRefl2SpectWhite.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBRefl2SpectWhite, N_RGB_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbRefl2SpectCyan.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBRefl2SpectCyan, N_RGB_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbRefl2SpectMagenta.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBRefl2SpectMagenta, N_RGB_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbRefl2SpectYellow.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBRefl2SpectYellow, N_RGB_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbRefl2SpectRed.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBRefl2SpectRed, N_RGB_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbRefl2SpectGreen.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBRefl2SpectGreen, N_RGB_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbRefl2SpectBlue.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBRefl2SpectBlue, N_RGB_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbIllum2SpectWhite.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectWhite, N_RGB_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbIllum2SpectCyan.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectCyan, N_RGB_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbIllum2SpectMagenta.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectMagenta, N_RGB_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbIllum2SpectYellow.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectYellow, N_RGB_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbIllum2SpectRed.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectRed, N_RGB_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbIllum2SpectGreen.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectGreen, N_RGB_SAMPLES, wl0[i], wl1[i])
			SampledSpectrum.rgbIllum2SpectBlue.c[i] = SampledSpectrum.average_spectrum_samples(RGB2SpectLambda, RGBIllum2SpectBlue, N_RGB_SAMPLES, wl0[i], wl1[i])


	def toXYZ(self):
		return np.array([
				np.sum(self.X.c * self.c),
				np.sum(self.Y.c * self.c),
				np.sum(self.Z.c * self.c)]) * (SAMPLED_LAMBDA_END - SAMPLED_LAMBDA_START) \
										   / (CIE_Y_INTEGRAL * N_SPECTRAL_SAMPLES)

	def y(self):
		return np.sum(self.Y.c * self.c) * (SAMPLED_LAMBDA_END - SAMPLED_LAMBDA_START) \
										 / (CIE_Y_INTEGRAL * N_SPECTRAL_SAMPLES)

	def toRGB(self) -> [FLOAT]:
		return xyz2rgb(self.toXYZ())

	def toRGBSepctrum(self) -> 'RGBSpectrum':
		return RGBSpectrum.fromXYZ(self.c)

class RGBSpectrum(CoefficientSpectrum):
	'''
	RGBSpectrum Class

	Subclasses `CoefficientSpectrum`, used to
	model RGB spectrum.
	'''
	def __init__(self, v: FLOAT = 0.):
		super().__init__(3, v)

	@staticmethod
	@jit
	def fromSampled(cls, lam: [FLOAT], v: [FLOAT], n: INT):
		ss = SampledSpectrum.fromSampled(lam, v, n)
		return cls.fromXYZ(ss.c)	

	@classmethod
	def fromCoefSpectrum(cls, s: 'CoefficientSpectrum'):
		if len(s.c) != 3:
			raise TypeError
		self = cls()
		self.c = s.c.copy()

	@staticmethod
	@jit
	def fromRGB(rgb: [FLOAT], tp: 'SpectrumType'):
		s = RGBSpectrum()
		s.c = np.array(rgb)
		return s

	@classmethod
	@jit
	def fromXYZ(cls, xyz: [FLOAT], tp: 'SpectrumType' = SpectrumType.REFLECTANCE):
		return cls.fromRGB(xyz2rgb(xyz), tp)

	def toRGB(self):
		return self.c.copy()

	def toRGBSpectrum(self):
		r = RGBSpectrum()
		r.c = self.c.copy()
		return r

	def toXYZ(self) -> [FLOAT]:
		return rgb2xyz(self.c)

	def y(self) -> FLOAT:
		y_weight = np.array([0.212671, 0.715160, 0.072169])
		return y_weight * self.c






