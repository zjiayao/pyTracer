'''
volume.py

Model volume scatterings.

Created by Jiayao on Aug 7, 2017
'''
from numba import jit
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
from src.core.pytracer import *
from src.core.geometry import *
from src.core.diffgeom import *
from src.core.spectrum import *
from src.core.reflection import *


# Global Utility Functions
## Phase Functions
@jit
def phase_isotrophic(w: 'Vector', wp: 'Vector') -> FLOAT:
	return 1. / (4. * PI

@jit
def phase_rayleigh(w: 'Vector', wp: 'Vector') -> FLOAT:
	ct = w.dot(wp)
	return 3. / (16. * PI) * (1. + ct * ct)

@jit
def phase_mie_hazy(w: 'Vector', wp: 'Vector') -> FLOAT:
	ct = w.dot(wp)
	return (.5 + 4.5 * np.power(.5 * (1. + ct), 8.)) / (4. * PI)
	
@jit
def phase_mie_murky(w: 'Vector', wp: 'Vector') -> FLOAT:
	ct = w.dot(wp)
	return (.5 + 16.5 * np.power(0.5 * (1. + ct), 32.)) / (4. * PI);

@jit
def phase_hg(w: 'Vector', wp: 'Vector', g: FLOAT) -> FLOAT:
	'''
	g: asymmetry parameter
	controls the distribution of light
	'''
	ct = w.dot(wp)
	return ((1. - g * g) / np.power(1. + g * g - 2. * g * ct, 1.5)) / (4. * PI);

@jit
def phase_schlick(w: 'Vector', wp: 'Vector', g: FLOAT) -> FLOAT:
	k = 1.55 * g - .55 * g * g * g
	kct = k * w.dot(wp)
	return ((1. - k * k) / (1. - kct) * (1. - kct)) / (4. * PI);

## Util Functions
@jit
def subsurface_from_diffuse(kd: 'Spectrum', mfp: FLOAT, eta: FLOAT): -> ['Spectrum', 'Spectrum']:
	'''
	Subsurface from Diffuse
	Returns:
	[sigma_a: 'Spectrum', sigma_prime_s: 'Spectrum']
	TODO
	'''
	pass

# Volume Interface

class VolumeRegion(object, metaclass=ABCMeta):
	'''
	VolumeRegion Class

	Models volume scattering in a
	region of the scene.
	'''
	def __init__(self):
		pass

	def __repr__(self):
		return "{}".format(self.__class__)


	@abstractmethod
	def world_bound(self) ->  'BBox':
		raise NotImplementedError('src.core.volume.{}.world_bound(): abstract method '
									'called'.format(self.__class__)) 		

	@abstractmethod
	def intersectP(self, ray: 'Ray') ->  [bool, FLOAT, FLOAT]:
		'''
		intersectP()

		Returns the parameter range of segment
		that overlaps the volume
		'''
		raise NotImplementedError('src.core.volume.{}.intersectP(): abstract method '
									'called'.format(self.__class__)) 

	@abstractmethod
	def sigma_a(self, pnt: 'Point', vec: 'Vector', tm: FLOAT) ->  'Spectrum':
		'''
		sigma_a()

		Given world point, direction and time.
		Returns absorption property.
		'''
		raise NotImplementedError('src.core.volume.{}.sigma_a(): abstract method '
									'called'.format(self.__class__)) 


	@abstractmethod
	def sigma_s(self, pnt: 'Point', vec: 'Vector', tm: FLOAT) ->  'Spectrum':
		'''
		sigma_s()

		Given world point, direction and time.
		Returns scattering property.
		'''
		raise NotImplementedError('src.core.volume.{}.sigma_s(): abstract method '
									'called'.format(self.__class__)) 

	@abstractmethod
	def lve(self, pnt: 'Point', vec: 'Vector', tm: FLOAT) ->  'Spectrum':
		'''
		lve()

		Given world point, direction and time.
		Returns emission property.
		'''
		raise NotImplementedError('src.core.volume.{}.lve(): abstract method '
									'called'.format(self.__class__)) 


	@abstractmethod
	def p(self, pnt: 'Point', wi: 'Vector', wo: 'Vector', tm: FLOAT) ->  FLOAT:
		'''
		p()

		Given world point, a pair of directions and time.
		Returns value of phase function.
		'''
		raise NotImplementedError('src.core.volume.{}.p(): abstract method '
									'called'.format(self.__class__)) 

	@abstractmethod
	def tau(self, ray: 'Ray', step: FLOAT=1., offset: FLOAT=.5) ->  'Spectrum:
		'''
		tau()

		Returns optical thickness within ray
		segment.
		'''
		raise NotImplementedError('src.core.volume.{}.tau(): abstract method '
									'called'.format(self.__class__)) 

	@property
	def sigma_t(self, pnt: 'Point', vec: 'Vector', tm: FLOAT) ->  'Spectrum':
		'''
		sigma_t()

		Given world point, direction and time.
		Returns attenuation coefficient, the sum
		of sigma_a and sigma_s by default.
		'''
		return self.sigma_a(p, w, tm) + self.sigma_s(p, w, tm)




class HomogeneousVolumeDensity(VolumeRegion):
	'''
	HomogeneousVolumeDensity Class

	Models a box-shaped region of space
	with homogeneous scattering properties.
	'''
	def __init__(self, sig_a: 'Spectrum', sig_s: 'Spectrum', g: FLOAT, le: 'Spectrum', 
					extent: 'BBox', v2w: 'Transform'):
		self.sig_a = sig_a
		self.sig_s = sig_s
		self.g = g
		self.le = le
		self.extent = extent
		self.w2v = v2w.inverse()	# note param is volume region to world transformation


	def world_bound(self) ->  'BBox':
		return self.w2v.inverse()(self.extent)


	def intersectP(self, ray: 'Ray') ->  [bool, FLOAT, FLOAT]:
		r = self.w2v(ray)
		return self.extent.intersectP(ray)

	def sigma_a(self, pnt: 'Point', vec: 'Vector', tm: FLOAT) ->  'Spectrum':
		return self.sig_a if self.extent.inside(self.w2v(pnt)) else Spectrum(0.)


	def sigma_s(self, pnt: 'Point', vec: 'Vector', tm: FLOAT) ->  'Spectrum':
		return self.sig_s if self.extent.inside(self.w2v(pnt)) else Spectrum(0.)

	def sigma_t(self, pnt: 'Point', vec: 'Vector', tm: FLOAT) ->  'Spectrum':
		return (self.sig_s + self.sig_a) if self.extent.inside(self.w2v(pnt)) else Spectrum(0.)		

	def lve(self, pnt: 'Point', vec: 'Vector', tm: FLOAT) ->  'Spectrum':
		return self.le if self.extent.inside(self.w2v(pnt)) else Spectrum(0.)

	def p(self, pnt: 'Point', wi: 'Vector', wo: 'Vector', tm: FLOAT) ->  FLOAT:
		if not self.extent.inside(self.w2v(p)):
			return 0.
		return phase_hg(wi, wo, g)

	def tau(self, ray: 'Ray', step: FLOAT=1., offset: FLOAT=.5) ->  'Spectrum:
		isec, t0, t1 = self.intersectP(ray)
		if isec:
			return (ray(t1) - ray(t0)).length() * (self.sig_a + self.sig_s)
		return Spectrum(0.)

# Density Region
class DensityRegion(VolumeRegion):
	'''
	DensityRegion Class

	Obtain the particle density
	'''
	def __init__(self, sig_a: 'Spectrum', sig_s: 'Spectrum', g: FLOAT, le: 'Spectrum',
					v2w: 'Transform'):
		self.sig_a = sig_a
		self.sig_s = sig_s
		self.g = g
		self.le = le
		self.w2v = v2w.inverse()

	@abstractmethod
	def __call__(self, pnt: 'Point') -> FLOAT:
		'''
		__call__()

		Returns the particle density at the given
		point in *object space*.
		Analogous to Density() in `pbrt`.
		'''
		raise NotImplementedError('src.core.volume.{}.__call__(): abstract method '
									'called'.format(self.__class__)) 		

	def sigma_a(self, pnt: 'Point', vec: 'Vector', tm: FLOAT) ->  'Spectrum':
		return self(self.w2v(p)) * self.sig_a


	def sigma_s(self, pnt: 'Point', vec: 'Vector', tm: FLOAT) ->  'Spectrum':
		return self(self.w2v(p)) * self.sig_s		
		

	def lve(self, pnt: 'Point', vec: 'Vector', tm: FLOAT) ->  'Spectrum':
		return self(self.w2v(p)) * self.le

	def p(self, pnt: 'Point', w: 'Vector', wp: 'Vector', tm: FLOAT) ->  FLOAT:
		'''
		p()

		Does not scale the phase function.
		'''
		return phase_hg(w, wp, g)

	@property
	def sigma_t(self, pnt: 'Point', vec: 'Vector', tm: FLOAT) ->  'Spectrum':
		return self(self.w2v(p)) * (self.sig_a + self.sig_s)


class VolumeGridDensity(DensityRegion):
	'''
	VolumeGridDensity Class

	Stores densities at regular 3D grid.
	'''
	def __init__(self, sig_a: 'Spectrum', sig_s: 'Spectrum', g: FLOAT, le: 'Spectrum',
					v2w: 'Transform', d: 'np.ndarray'):
		self.sig_a = sig_a
		self.sig_s = sig_s
		self.g = g
		self.le = le
		self.w2v = v2w.inverse()
		self.density = d.copy()	# 3D np array of densities in [z][y][x] order

	@property
	def nx(self):
		return np.shape(self.density)[0]

	@property
	def ny(self):
		return np.shape(self.density)[1]

	@property
	def nz(self):
		return np.shape(self.density)[2]

	@jit
	def __D(self, x: INT, y: INT, z: INT) -> FLOAT:
		x, y, z = np.clip([x, y, z], 0, np.shape(self.density) - 1).astype(INT)
		return self.density[z][y][x]

	@jit
	def __call__(self, pnt: 'Point') -> FLOAT:
		if not self.extent.inside(pnt):
			return 0.

		# compute voxel coordinates and offsets
		vox = self.extent.offset(pnt)
		vox *= np.shape(self.density)
		vox -= .5
		vx = ftoi(vox.x)
		vy = ftoi(vox.y)
		vz = ftoi(vox.z)
		dx = vox.x - vx
		dy = vox.y - vy
		dz = vox.z - vz

		# trilinearly interpolation
		d00 = Lerp(dx, self.__D(vx, vy, vz), self.__D(vx+1,vy,vz))
		d10 = Lerp(dx, self.__D(vx, vy+1, vz), self.__D(vx+1,vy+1,vz))
		d01 = Lerp(dx, self.__D(vx, vy, vz+1), self.__D(vx+1,vy,vz+1))
		d11 = Lerp(dx, self.__D(vx, vy+1, vz+1), self.__D(vx+1,vy+1,vz+1))

		d0 = Lerp(dy, d00, d10)
		d1 = Lerp(dy, d01, d11)

		return Lerp(dz, d0, d1)


	def world_bound(self) ->  'BBox':
		return self.w2v.inverse()(self.extent)


	def intersectP(self, ray: 'Ray') ->  [bool, FLOAT, FLOAT]:
		r = self.w2v(ray)
		return self.extent.intersectP(ray)

class ExponentialDensity(DensityRegion):
	'''
	ExponentialDensity Class

	Exponentially decaying densities
	'''
	def __init__(self, sig_a: 'Spectrum', sig_s: 'Spectrum', g: FLOAT, le: 'Spectrum',
				extent: 'BBox', v2w: 'Transform', a: FLOAT, b: FLOAT, up: 'Vector'):
		super().__init__(sig_a, sig_s, g, le, v2w)
		self.extent = extent
		self.a = a
		self.b = b
		self.up = normalize(up)

	@jit
	def __call__(self, pnt: 'Point') -> FLOAT:
		if not self.extent.inside(pnt):
			return 0.
		h = (pnt - self.extent.pMin).dot(self.up)
		return self.a * np.exp(-self.b * h)

	def world_bound(self) ->  'BBox':
		return self.w2v.inverse()(self.extent)


	def intersectP(self, ray: 'Ray') ->  [bool, FLOAT, FLOAT]:
		r = self.w2v(ray)
		return self.extent.intersectP(ray)

class AggregateVolume(VolumeRegion):
	'''
	AggregateVolume Class

	Aggregate class for `VolumeRegion`.
	By default uses inefficient implementation
	'''
	def __init__(self, regions: list):
		self.regions = regions
		self.bound = BBox()
		for region in regions:
			self.bound.union(region.world_bound())

	def sigma_a(self, pnt: 'Point', w: 'Vector', tm: FLOAT) ->  'Spectrum':
		s = Spectrum(0.)
		for region in self.regions:
			s += region.sigma_a(pnt, w, tm)
		return s

	def sigma_s(self, pnt: 'Point', w: 'Vector', tm: FLOAT) ->  'Spectrum':
		s = Spectrum(0.)
		for region in self.regions:
			s += region.sigma_s(pnt, w, tm)
		return s

	def sigma_t(self, pnt: 'Point', w: 'Vector', tm: FLOAT) ->  'Spectrum':
		s = Spectrum(0.)
		for region in self.regions:
			s += region.sigma_t(pnt, w, tm)
		return s

	def lve(self, pnt: 'Point', w: 'Vector', tm: FLOAT) ->  'Spectrum':
		s = Spectrum(0.)
		for region in self.regions:
			s += region.le(pnt, w, tm)
		return s		

	def p(self, pnt: 'Point', wi: 'Vector', wo: 'Vector', tm: FLOAT) ->  FLOAT:
		s = Spectrum(0.)
		for region in self.regions:
			s += region.p(pnt, wi, wo, tm)
		return s		

	def tau(self, ray: 'Ray', step: FLOAT=1., offset: FLOAT=.5) ->  'Spectrum:
		s = Spectrum(0.)
		for region in self.regions:
			s += region.tau(ray, step, offset)
		return s	

	def world_bound(self) ->  'BBox':
		return self.w2v.inverse()(self.bound)


	def intersectP(self, ray: 'Ray') ->  [bool, FLOAT, FLOAT]:
		t0 = np.inf
		t1 = -np.inf
		for region in self.regions:
			isect, r0, r1 = region.intersectP(ray)
			if isect:
				t0 = min(t0, r0)
				t1 = max(t1, r1)
		return [(t0 < t1), t0, t1]

	

# BSSRDF
class BSSRDF(object):
	'''
	BSSRDF Class

	Models low-level scattering properties.
	Scattering computed by integrators.
	'''
	def __init__(self, sig_a: 'Spectrum', sigp_s: 'Spectrum', e: FLOAT):
		self.__e = e
		self.__sig_a = sig_a
		self.__sigp_s = sigp_s #reduced scattering coefficient (1. - sigma_s)

	def __repr__(self):
		return "{}\nEta: {} sigma_a: {} sigma'_s: {}\n".format(self.__class__,
						self.__sig_a, self.__sigp_s)

	@property
	def eta(self):
		return self.__e

	@property
	def sigma_a(self):
		return self.__sig_a

	@property
	def sigma_prime_s(self):
		return self.__sigp_s


# measured data
from src.data.volume import MEASURED_SUF_SC
def get_volume_scattering(name: str) -> ['Spectrum', 'Spectrum']:
	'''
	Load volume scattering data,
	returns `Spectrum`s of
	[sigma_a, sigma'_s]
	'''
	if name in MEASURED_SUF_SC:
		return [Spectrum.fromRGB(MEASURED_SUF_SC[name][0]),
				Spectrum.fromRGB(MEASURED_SUF_SC[name][1])]
	return [None, None]































