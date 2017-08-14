"""
volume.py

Model volume scatterings.

Created by Jiayao on Aug 7, 2017
"""
from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
from pytracer import *
import pytracer.geometry as geo
import pytracer.transform as trans

__all__ = ['VolumeRegion', 'DensityRegion', 'HomogeneousVolumeDensity',
           'VolumeGridDensity', 'ExponentialDensity', 'AggregateVolume']


# Volume Interface
class VolumeRegion(object, metaclass=ABCMeta):
	"""
	VolumeRegion Class

	Models volume scattering in a
	region of the scene.
	"""
	def __init__(self):
		pass

	def __repr__(self):
		return "{}".format(self.__class__)


	@abstractmethod
	def world_bound(self) ->  'geo.BBox':
		raise NotImplementedError('src.core.volume.{}.world_bound(): abstract method '
									'called'.format(self.__class__)) 		

	@abstractmethod
	def intersect_p(self, ray: 'geo.Ray') ->  [bool, FLOAT, FLOAT]:
		"""
		intersect_p()

		Returns the parameter range of segment
		that overlaps the volume
		"""
		raise NotImplementedError('src.core.volume.{}.intersect_p(): abstract method '
									'called'.format(self.__class__)) 

	@abstractmethod
	def sigma_a(self, pnt: 'geo.Point', vec: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		"""
		sigma_a()

		Given world point, direction and time.
		Returns absorption property.
		"""
		raise NotImplementedError('src.core.volume.{}.sigma_a(): abstract method '
									'called'.format(self.__class__)) 


	@abstractmethod
	def sigma_s(self, pnt: 'geo.Point', vec: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		"""
		sigma_s()

		Given world point, direction and time.
		Returns scattering property.
		"""
		raise NotImplementedError('src.core.volume.{}.sigma_s(): abstract method '
									'called'.format(self.__class__)) 

	@abstractmethod
	def lve(self, pnt: 'geo.Point', vec: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		"""
		lve()

		Given world point, direction and time.
		Returns emission property.
		"""
		raise NotImplementedError('src.core.volume.{}.lve(): abstract method '
									'called'.format(self.__class__)) 


	@abstractmethod
	def p(self, pnt: 'geo.Point', wi: 'geo.Vector', wo: 'geo.Vector', tm: FLOAT) ->  FLOAT:
		"""
		p()

		Given world point, a pair of directions and time.
		Returns value of phase function.
		"""
		raise NotImplementedError('src.core.volume.{}.p(): abstract method '
									'called'.format(self.__class__)) 

	@abstractmethod
	def tau(self, ray: 'geo.Ray', step: FLOAT=1., offset: FLOAT=.5) ->  'Spectrum':
		"""
		tau()

		Returns optical thickness within ray
		segment.
		"""
		raise NotImplementedError('src.core.volume.{}.tau(): abstract method '
									'called'.format(self.__class__)) 

	def sigma_t(self, pnt: 'geo.Point', vec: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		"""
		sigma_t()

		Given world point, direction and time.
		Returns attenuation coefficient, the sum
		of sigma_a and sigma_s by default.
		"""
		return self.sigma_a(pnt, vec, tm) + self.sigma_s(pnt, vec, tm)


class HomogeneousVolumeDensity(VolumeRegion):
	"""
	HomogeneousVolumeDensity Class

	Models a box-shaped region of space
	with homogeneous scattering properties.
	"""
	def __init__(self, sig_a: 'Spectrum', sig_s: 'Spectrum', g: FLOAT, le: 'Spectrum', 
					extent: 'geo.BBox', v2w: 'trans.Transform'):
		super().__init__()
		self.sig_a = sig_a
		self.sig_s = sig_s
		self.g = g
		self.le = le
		self.extent = extent
		self.w2v = v2w.inverse()	# note param is volume region to world transformation


	def world_bound(self) ->  'geo.BBox':
		return self.w2v.inverse()(self.extent)


	def intersect_p(self, ray: 'geo.Ray') ->  [bool, FLOAT, FLOAT]:
		r = self.w2v(ray)
		return self.extent.intersect_p(ray)

	def sigma_a(self, pnt: 'geo.Point', vec: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		return self.sig_a if self.extent.inside(self.w2v(pnt)) else Spectrum(0.)


	def sigma_s(self, pnt: 'geo.Point', vec: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		return self.sig_s if self.extent.inside(self.w2v(pnt)) else Spectrum(0.)

	def sigma_t(self, pnt: 'geo.Point', vec: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		return (self.sig_s + self.sig_a) if self.extent.inside(self.w2v(pnt)) else Spectrum(0.)		

	def lve(self, pnt: 'geo.Point', vec: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		return self.le if self.extent.inside(self.w2v(pnt)) else Spectrum(0.)

	def p(self, pnt: 'geo.Point', wi: 'geo.Vector', wo: 'geo.Vector', tm: FLOAT) ->  FLOAT:
		if not self.extent.inside(self.w2v(pnt)):
			return 0.
		from pytracer.volume.utility import phase_hg
		return phase_hg(wi, wo, tm)

	def tau(self, ray: 'geo.Ray', step: FLOAT=1., offset: FLOAT=.5) ->  'Spectrum':
		isec, t0, t1 = self.intersect_p(ray)
		if isec:
			return (ray(t1) - ray(t0)).length() * (self.sig_a + self.sig_s)
		return Spectrum(0.)


# Density Region
class DensityRegion(VolumeRegion):
	"""
	DensityRegion Class

	Obtain the particle density
	"""
	def __init__(self, sig_a: 'Spectrum', sig_s: 'Spectrum', g: FLOAT, le: 'Spectrum',
					v2w: 'trans.Transform'):
		super().__init__()
		self.sig_a = sig_a
		self.sig_s = sig_s
		self.g = g
		self.le = le
		self.w2v = v2w.inverse()

	@abstractmethod
	def __call__(self, pnt: 'geo.Point') -> FLOAT:
		"""
		__call__()

		Returns the particle density at the given
		point in *object space*.
		Analogous to Density() in `pbrt`.
		"""
		raise NotImplementedError('src.core.volume.{}.__call__(): abstract method '
									'called'.format(self.__class__)) 		

	def sigma_a(self, pnt: 'geo.Point', vec: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		return self(self.w2v(pnt)) * self.sig_a

	def sigma_s(self, pnt: 'geo.Point', vec: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		return self.sig_s * self(self.w2v(pnt))

	def lve(self, pnt: 'geo.Point', vec: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		return self(self.w2v(pnt)) * self.le

	def p(self, pnt: 'geo.Point', w: 'geo.Vector', wp: 'geo.Vector', tm: FLOAT) ->  FLOAT:
		"""
		p()

		Does not scale the phase function.
		"""
		from pytracer.volume.utility import phase_hg
		return phase_hg(w, wp, tm)

	def sigma_t(self, pnt: 'geo.Point', vec: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		return self(self.w2v(pnt)) * (self.sig_a + self.sig_s)

	def tau(self, r: 'geo.Ray', step_size: FLOAT, u: FLOAT) -> 'Spectrum':
		tau = Spectrum(0.
			)
		length = r.d.length()
		if length == 0.:
			return tau

		rn = geo.Ray(r.o, r.d / length, r.mint * length, r.maxt * length, r.time)
		hit, t0, t1 = self.intersect_p(rn)
		if not hit:
			return tau

		t0 += u * step_size
		while t0 < t1:
			tau += self.sigma_t(rn(t0), -rn.d, r.time)
			t0 += step_size

		return tau * step_size


class VolumeGridDensity(DensityRegion):
	"""
	VolumeGridDensity Class

	Stores densities at regular 3D grid.
	"""
	def __init__(self, sig_a: 'Spectrum', sig_s: 'Spectrum', g: FLOAT, le: 'Spectrum',
					extent: 'geo.BBox', v2w: 'trans.Transform', d: 'np.ndarray'):
		super().__init__()
		self.sig_a = sig_a
		self.sig_s = sig_s
		self.g = g
		self.le = le
		self.extent = extent
		self.w2v = v2w.inverse()
		self.density = d.copy() 	# 3D np array of densities in [z][y][x] order

	@property
	def nx(self):
		return np.shape(self.density)[0]

	@property
	def ny(self):
		return np.shape(self.density)[1]

	@property
	def nz(self):
		return np.shape(self.density)[2]

	def __D(self, x: INT, y: INT, z: INT) -> FLOAT:
		x, y, z = np.clip([x, y, z], 0, np.shape(self.density) - 1).astype(INT)
		return self.density[z][y][x]
	
	def __call__(self, pnt: 'geo.Point') -> FLOAT:
		if not self.extent.inside(pnt):
			return 0.

		# compute voxel coordinates and offsets
		vox = self.extent.offset(pnt)
		vox *= np.shape(self.density)
		vox -= .5
		vx = util.ftoi(vox.x)
		vy = util.ftoi(vox.y)
		vz = util.ftoi(vox.z)
		dx = vox.x - vx
		dy = vox.y - vy
		dz = vox.z - vz

		# trilinearly interpolation
		d00 = util.lerp(dx, self.__D(vx, vy, vz), self.__D(vx+1,vy,vz))
		d10 = util.lerp(dx, self.__D(vx, vy+1, vz), self.__D(vx+1,vy+1,vz))
		d01 = util.lerp(dx, self.__D(vx, vy, vz+1), self.__D(vx+1,vy,vz+1))
		d11 = util.lerp(dx, self.__D(vx, vy+1, vz+1), self.__D(vx+1,vy+1,vz+1))

		d0 = util.lerp(dy, d00, d10)
		d1 = util.lerp(dy, d01, d11)

		return util.lerp(dz, d0, d1)


	def world_bound(self) ->  'geo.BBox':
		return self.w2v.inverse()(self.extent)


	def intersect_p(self, ray: 'geo.Ray') ->  [bool, FLOAT, FLOAT]:
		r = self.w2v(ray)
		return self.extent.intersect_p(ray)


class ExponentialDensity(DensityRegion):
	"""
	ExponentialDensity Class

	Exponentially decaying densities
	"""
	def __init__(self, sig_a: 'Spectrum', sig_s: 'Spectrum', g: FLOAT, le: 'Spectrum',
				extent: 'geo.BBox', v2w: 'trans.Transform', a: FLOAT, b: FLOAT, up: 'geo.Vector'):
		super().__init__(sig_a, sig_s, g, le, v2w)
		self.extent = extent
		self.a = a
		self.b = b
		self.up = geo.normalize(up)

	
	def __call__(self, pnt: 'geo.Point') -> FLOAT:
		if not self.extent.inside(pnt):
			return 0.
		h = (pnt - self.extent.pMin).dot(self.up)
		return self.a * np.exp(-self.b * h)

	def world_bound(self) ->  'geo.BBox':
		return self.w2v.inverse()(self.extent)


	def intersect_p(self, ray: 'geo.Ray') ->  [bool, FLOAT, FLOAT]:
		r = self.w2v(ray)
		return self.extent.intersect_p(ray)


class AggregateVolume(VolumeRegion):
	"""
	AggregateVolume Class

	Aggregate class for `VolumeRegion`.
	By default uses inefficient implementation
	"""
	def __init__(self, regions: list):
		self.regions = regions
		self.bound = geo.BBox()
		for region in regions:
			self.bound.union(region.world_bound())

	def sigma_a(self, pnt: 'geo.Point', w: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		s = Spectrum(0.)
		for region in self.regions:
			s += region.sigma_a(pnt, w, tm)
		return s

	def sigma_s(self, pnt: 'geo.Point', w: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		s = Spectrum(0.)
		for region in self.regions:
			s += region.sigma_s(pnt, w, tm)
		return s

	def sigma_t(self, pnt: 'geo.Point', w: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		s = Spectrum(0.)
		for region in self.regions:
			s += region.sigma_t(pnt, w, tm)
		return s

	def lve(self, pnt: 'geo.Point', w: 'geo.Vector', tm: FLOAT) ->  'Spectrum':
		s = Spectrum(0.)
		for region in self.regions:
			s += region.le(pnt, w, tm)
		return s		

	def p(self, pnt: 'geo.Point', wi: 'geo.Vector', wo: 'geo.Vector', tm: FLOAT) ->  FLOAT:
		s = Spectrum(0.)
		for region in self.regions:
			s += region.p(pnt, wi, wo, tm)
		return s		

	def tau(self, ray: 'geo.Ray', step: FLOAT=1., offset: FLOAT=.5) ->  'Spectrum':
		s = Spectrum(0.)
		for region in self.regions:
			s += region.tau(ray, step, offset)
		return s	

	def world_bound(self) ->  'geo.BBox':
		return self.bound
		#return self.w2v.inverse()(self.bound)

	def intersect_p(self, ray: 'geo.Ray') ->  [bool, FLOAT, FLOAT]:
		t0 = np.inf
		t1 = -np.inf
		for region in self.regions:
			isect, r0, r1 = region.intersect_p(ray)
			if isect:
				t0 = min(t0, r0)
				t1 = max(t1, r1)
		return [(t0 < t1), t0, t1]



























