'''
light.py

Model physically plausible lights.

Created by Jiayao on Aug 8, 2017
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
from src.core.texture import *

# Utility Classes

class VisibilityTester(object):
	'''
	VisibilityTester Class
	'''
	def __init__(self, ray: 'Ray'=None):
		self.ray = None

	def __repr__(self):
		return "{}\nRay: {}\n".format(self.__class__, self.ray)

	def set_segment(self, p1: 'Point', eps1: FLOAT, p2: 'Point', eps2: FLOAT, time: FLOAT):
		'''
		set_segment()

		The test is to be done within the
		given segment.
		'''
		dist = (p1 - p2).length()
		self.ray = Ray(p1, (p2 - p1) / dist, eps1, dist * (1. - eps2), time)

	def set_ray(self, p: 'Point', eps: FLOAT, w: 'Vector', time: FLOAT):
		'''
		set_ray()

		The test is to indicate whether there
		is any object along a given direction.
		'''
		self.ray = Ray(p, w, eps, np.inf, time)

	def unoccluded(self, scene: 'Scene') -> bool:
		'''
		unoccluded()

		Traces a shadow ray
		'''
		return not scene.intersectP(self.ray)

	def transmittance(self, scene: 'Scene', renderer: 'Renderer', sample: 'Sample',
						rng=np.random.rand):
		'''
		transmittance()

		Determines the fraction of illumincation from
		the light to the point that is not extinguished
		by participating media.
		'''
		return renderer.transmittance(scene, RayDifferential.fromRay(self.ray), sample. rng)


def ShapeSet(object):
	'''
	ShapeSet Class

	Wrapper for a set of `Shape`s.
	'''
	def __init__(self, shape: 'Shape'):
		self.shapes = []
		tmp = [s]
		while len(tmp) > 0:
			sh = tmp.pop()
			if sh.can_intersect():
				shapes.append(sh)
			else:
				tmp.extend(sh.refine())

		if len(self.shapes) > 64:
			print("Warning: src.core.light.{}.__init__(): "
					"Area light turned into {} shapes, might be inefficient".format(self.__class__,
							len(self.shapes)))
		self.areas = []
		self.sum_area = 0.
		for sh in self.shapes:
			area = sh.area()
			self.areas.push_back(area)
			self.sum_area += area

		# TODO
		# area_distribution





# Light Classes

class Light(object, metaclass=ABCMeta):
	'''
	Light Class
	'''
	def __init__(self, l2w: 'Transform', ns: INT = 1):
		'''
		l2w: Light-to-World `Transform`
		ns:  number of samples for soft shadowing of
		 	 area light
		'''
		self.ns = max(1, ns)
		self.l2w = l2w
		self.w2l = l2w.inverse()

		if l2w.has_scale():
			print('Warning: src.core.light.{}.__init__() light '
					'transforms should not contain scale'.format(self.__class__))

	def __repr__(self):
		return "{}\nNumber of Samples: {}\nLight-to-World "
					"Transformation: {}\n".format(self.__class__, self.ns, self.l2w)

	@abstractmethod
	def sample_l(self, p: 'Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'Vector', FLOAT, 'VisibilityTester']:	
		'''
		sample_l()

		Returns the radiance arriving at a
		given point, the incident direction, 
		pdf is used in MC sampling, i.e.,
		[Spectrum, wi, pdf, tester]
		'''
		raise NotImplementedError('src.core.light.{}.sample_l(): abstract method '
									'called'.format(self.__class__)) 		

	@abstractmethod
	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		'''
		power()

		Returns the total emitted power
		of the light.
		'''
		raise NotImplementedError('src.core.light.{}.power(): abstract method '
									'called'.format(self.__class__)) 	

	@abstractmethod
	def is_delta_light(self) -> bool:
		'''
		is_delta_light()

		Determines whether the light
		follows delta distribution, e.g.,
		point light, spotlight et.c.
		'''
		raise NotImplementedError('src.core.light.{}.is_delta_light(): abstract method '
									'called'.format(self.__class__)) 	

	def le(self, rd: 'RayDifferential') -> 'Spectrum':
		'''
		le()

		Returns emitted radiance along
		a ray hit nothing
		'''
		return Spectrum(0.)


class PointLight(Light):
	'''
	PointLight Class
	
	By defauly positioned at the origin.
	'''
	def __init__(self, l2w: 'Transform', intensity: 'Spectrum'):
		super().__init__(l2w)
		self.pos = l2w(Point(0., 0., 0.)) # where the light is positioned in the world
		self.intensity = intensity


	def sample_l(self, p: 'Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'Vector', FLOAT, 'VisibilityTester']:	
		wi = normalize(self.pos - p)
		pdf = 1.
		vis = VisibilityTester()
		vis.set_segment(p, pEps, self.pos, 0., time)
		return [self.intensity / (self.pos - p).sq_length(), wi, pdf, vis]

	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		return 4. * PI * self.intensity

	def is_delta_light(self) -> bool:
		return True



class SpotLight(Light):
	'''
	SpotLight Class
	
	By defauly positioned at the origin
	and light a cone towards +z.
	'''
	def __init__(self, l2w: 'Transform', intensity: 'Spectrum', width: FLOAT, falloff: FLOAT):
		'''
		width: Overall angular width of the cone
		fall: angle at which falloff starts
		'''
		super().__init__(l2w)
		self.pos = l2w(Point(0., 0., 0.)) # where the light is positioned in the world
		self.intensity = intensity
		self.cos_width = np.cos(np.deg2rad(width))
		self.cos_falloff = np.cos(np.deg2rad(falloff))

	@jit
	def __falloff(self, w: 'Vector') -> FLOAT:
		'''
		__falloff()

		Determines the falloff
		given a vector in the world.
		'''
		wl = normalize(self.w2l(w))
		ct = wl.z
		if ct < self.cos_width:
			return 0.
		if ct > self.cos_falloff:
			return 1.
		# falloff inside the cone
		d = (ct - self.cos_width) / (self.cos_falloff - self.cos_width)
		return d * d * d * d


	def sample_l(self, p: 'Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'Vector', FLOAT, 'VisibilityTester']:	
		wi = normalize(self.pos - p)
		pdf = 1.
		vis = VisibilityTester()
		vis.set_segment(p, pEps, self.pos, 0., time)
		return [self.intensity / self.__falloff(-wi), wi, pdf, vis]

	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		'''
		power()

		approximate by the integral over
		spread angle cosine halfway
		between width and falloff
		'''
		return self.intensity * 2. * PI * (1. - .5 * (self.cos_falloff + self.cos_width))

	def is_delta_light(self) -> bool:
		return True



class ProjectionLight(Light):
	'''
	ProjectionLight Class
	
	Projecting light using texture.
	'''
	def __init__(self, l2w: 'Transform', intensity: 'Spectrum', texname: str, fov: FLOAT):
		super().__init__(l2w)
		self.pos = l2w(Point(0., 0., 0.)) # where the light is positioned in the world
		self.intensity = intensity

		# create a MIPMap
		try:
			texels = read_image(filename)
			width, height = np.shape(texels)
		except:
			print('src.core.texture.{}.get_texture(): cannot process file {}, '
					'use default one-valued MIPMap'.format(self.__class__, filename))
			texels = None		
		
		if texels is not None:
			self.projMap = MIPMap(RGBSpectrum, texels)
		else:
			self.projMap = None

		# init projection matrix
		aspect = width / height
		if aspect > 1.:
			self.scr_x0 = -aspect
			self.scr_x1 = aspect
			self.scr_y0 = -1.
			self.scr_y1 = 1.
		else:
			self.scr_x0 = -1.
			self.scr_x1 = 1.
			self.scr_y0 = -1. / aspect
			self.scr_y1 = 1. / aspect

		self.hither = EPS
		self.yon = 1e30
		self.projTrans = Transform.perspective(fov, hither, yon)

		# compute cosine of cone
		self.cos_width = np.cos(np.arctan(np.tan(np.deg2rad(fov) / 2.) * np.sqrt(1. + 1. / (aspect * aspect))))


	def __projection(self, w: 'Vector') -> 'Spectrum':
		'''
		__projection()

		Utility method to determine
		the amount of light projected
		in the given direction.
		'''
		wl = self.w2l(w)
		# discard directions behind proj light
		if wl.z < self.hither:
			return Spectrum(0.)

		# project point onto plane
		pl = self.projTrans(Point(wl.x, wl.y, wl.z))
		if pl.x < self.scr_x0 or pl.x > scr_x1 or \
				pl.y < self.scr_y0 or py.y > scr_y1:
			return Spectrum(0.)
		if self.projMap is None:
			return Spectrum(1.)

		s = (pl.x - self.scr_x0) / (self.scr_x1 - self.scr_x0)
		t = (pl.y - self.scr_y0) / (self.scr_y1 - self.scr_y0)
		return Spectrum(self.projMap.look_up([s, t]), SpectrumType.ILLUMINANT) 

	def sample_l(self, p: 'Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'Vector', FLOAT, 'VisibilityTester']:	
		wi = normalize(self.pos - p)
		pdf = 1.
		vis = VisibilityTester()
		vis.set_segment(p, pEps, self.pos, 0., time)
		return [self.intensity / self.__projection(-wi), wi, pdf, vis]

	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		'''
		power()

		first order approximate by the 
		diagonal of the image, scaled
		by the average intensity.
		'''
		return Spectrum( self.projMap.look_up([.5, .5, .5]), SpectrumType.ILLUMINANT) * \
					self.intensity * 2. * PI * (1. - self.cos_width)

	def is_delta_light(self) -> bool:
		return True



class GonioPhotometricLight(Light):
	'''
	GonioPhotometricLight Class
	
	Projecting light using texture.
	'''
	def __init__(self, l2w: 'Transform', intensity: 'Spectrum', texname: str):
		super().__init__(l2w)
		self.pos = l2w(Point(0., 0., 0.)) # where the light is positioned in the world
		self.intensity = intensity

		# create a MIPMap
		try:
			texels = read_image(filename)
			width, height = np.shape(texels)
		except:
			print('src.core.texture.{}.get_texture(): cannot process file {}, '
					'use default one-valued MIPMap'.format(self.__class__, filename))
			texels = None		
		
		if texels is not None:
			self.mipmap = MIPMap(RGBSpectrum, texels)
		else:
			self.mipmap = None


	def __scale(self, w: 'Vector') -> 'Spectrum':
		'''
		__scale()

		Utility method to scale
		the amount of light projected
		in the given direction. Assume
		the scale texture is encoded
		using spherical coordinates.
		'''
		if self.mipmap is None:
			return Spectrum(1.)

		wp = normalize(self.w2l(w))
		wp.z, wp.y = wp.y, wp.z
		theta = spherical_theta(wp)
		phi = spherical_phi(wp)
		s = phi * INV_2PI
		t = theta * INV_PI

		return Spectrum(self.projMap.look_up([s, t]), SpectrumType.ILLUMINANT)

	def sample_l(self, p: 'Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'Vector', FLOAT, 'VisibilityTester']:	
		wi = normalize(self.pos - p)
		pdf = 1.
		vis = VisibilityTester()
		vis.set_segment(p, pEps, self.pos, 0., time)
		return [self.intensity / self.__scale(-wi), wi, pdf, vis]

	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		retrun 4. * PI * self.intensity * \
					Spectrum(self.mipmap.look_up([.5, .5, .5]), SpectrumType.ILLUMINANT)					
	def is_delta_light(self) -> bool:
		return True


class DistantLight(Light):
	'''
	DistantLight Class
	
	Modelling distant or directional
	light, i.e., point light
	at infinity.
	'''
	def __init__(self, l2w: 'Transform', radiance: 'Spectrum', di: 'Vector'):
		super().__init__(l2w)
		self.di = normalize(l2w(di))
		self.l = radiance

	def sample_l(self, p: 'Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'Vector', FLOAT, 'VisibilityTester']:	
		wi = self.di.copy()
		pdf = 1.
		vis = VisibilityTester()
		vis.set_ray(p, pEps, self.pos, wi, time)
		return [self.l.copy(), wi, pdf, vis]

	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		'''
		power()

		Approximate using a disk inside
		the scene's bounding sphere.
		'''
		_, rad = scene.world_bound().bounding_sphere()
	
		retrun self.l * PI * rad * rad	

	def is_delta_light(self) -> bool:
		return True


# Interface for Area Light
class AreaLight(Light):
	'''
	AreaLight Class
	'''
	def __init__(self, l2w: 'Transform', ns: INT):
		super().__init__(l2w, ns)
	@abstractmethod
	def l(self, p: 'Point', n: 'Normal', w: 'Vector') -> 'Spectrum':
		raise NotImplementedError('src.core.volume.{}.l(): abstract method '
									'called'.format(self.__class__)) 		




class DiffuseAreaLight(Light):
	'''
	DiffuseAreaLight Class
	
	Basic area light source with a
	uniform spatial and directional
	radiance distribution.
	'''
	def __init__(self, l2w: 'Transform', le: 'Spectrum', ns: INT, shape: 'Shape'):
		'''
		shape parameter will be a `ShapeSet`
		instance which subclasses `Shape`
		for easy implementation.
		'''
		super().__init__(l2w, ns)
		self.le = le
		self.shape_set = ShapeSet(s)
		self.area = self.shape_set.area()

	def l(self, p: 'Point', n: 'Normal', w: 'Vector') -> 'Spectrum':
		return self.le if n.dot(w) > 0. else Spectrum(0.)

	# TODO use MCMC
	# def sample_l(self, p: 'Point', pEps: FLOAT, ls: 'LightSample',
	# 		time: FLOAT,) -> ['Spectrum', 'Vector', FLOAT, 'VisibilityTester']:	

	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		return self.le * self.area * PI

	def is_delta_light(self) -> bool:
		return False




class PointLight(Light):
	'''
	PointLight Class
	
	By defauly positioned at the origin.
	'''
	def __init__(self, l2w: 'Transform', intensity: 'Spectrum'):
		super().__init__(l2w)
		self.pos = l2w(Point(0., 0., 0.)) # where the light is positioned in the world
		self.intensity = intensity


	def sample_l(self, p: 'Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'Vector', FLOAT, 'VisibilityTester']:	
		wi = normalize(self.pos - p)
		pdf = 1.
		vis = VisibilityTester()
		vis.set_segment(p, pEps, self.pos, 0., time)
		return [self.intensity / (self.pos - p).sq_length(), wi, pdf, vis]

	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		return 4. * PI * self.intensity

	def is_delta_light(self) -> bool:
		return True



class InfiniteAreaLight(Light):
	'''
	InfiniteAreaLight Class
	
	Models environment lighting
	'''
	def __init__(self, l2w: 'Transform', l: 'Spectrum', ns: INT, texmap: str):
		'''
		width: Overall angular width of the cone
		fall: angle at which falloff starts
		'''
		super().__init__(l2w, ns)
		self.pos = l2w(Point(0., 0., 0.)) # where the light is positioned in the world
		
		try:
			texels = read_image(filename)
			width, height = np.shape(texels)
		except:
			print('src.core.texture.{}.get_texture(): cannot process file {}, '
					'use default one-valued MIPMap'.format(self.__class__, filename))
			texels = None		
		
		if texels is not None:
			self.radMap = MIPMap(RGBSpectrum, texels)
		else:
			self.radMap = None

		# init sampling PDFs <725>

	def le(self, rd: 'RayDifferential') -> 'Spectrum':
		wh = normalize(self.w2l(rd.d))
		s = spherical_phi(wh) * INV_2PI
		t = spherical_theta(wh) * INV_PI
		return Spectrum(self.radMap.look_up(s, t), SpectrumType.ILLUMINANT)		

	# TODO use MCMC
	# def sample_l(self, p: 'Point', pEps: FLOAT, ls: 'LightSample',
	# 		time: FLOAT,) -> ['Spectrum', 'Vector', FLOAT, 'VisibilityTester']:	


	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		_, rad = scene.world_bound().bounding_sphere()
		retrun  PI * rad * rad	* Spectrum(self.radMap.look_up([.5, .5, .5], SpectrumType.ILLUMINANT)

	def is_delta_light(self) -> bool:
		return False






















