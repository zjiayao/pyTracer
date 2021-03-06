"""
light.py

Model physically plausible lights.

Created by Jiayao on Aug 8, 2017
"""

from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
from pytracer import *
import pytracer.geometry as geo
import pytracer.transform as trans
import pytracer.montecarlo as mc
from pytracer.light.utility import *
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from pytracer.scene import Scene



__all__ = ['Light', 'PointLight', 'SpotLight', 'ProjectionLight',
           'GonioPhotometricLight', 'DistantLight', 'AreaLight',
           'DiffuseAreaLight', 'InfiniteAreaLight']


# Light Classes
class Light(object, metaclass=ABCMeta):
	"""
	Light Class
	"""
	def __init__(self, l2w: 'trans.Transform', ns: INT = 1):
		"""
		l2w: Light-to-World `trans.Transform`
		ns:  number of samples for soft shadowing of
		 	 area light
		"""
		self.ns = max(1, ns)
		self.l2w = l2w
		self.w2l = l2w.inverse()

		if l2w.has_scale():
			print('Warning: src.core.light.{}.__init__() light '
					'trans.Transforms should not contain scale'.format(self.__class__))

	def __repr__(self):
		return '{}\nNumber of Samples: {}\nLight-to-World trans.Transformation: {}\n'.format(self.__class__, self.ns, self.l2w)

	@abstractmethod
	def sample_l(self, p: 'geo.Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT) -> ['Spectrum', 'geo.Vector', FLOAT, 'VisibilityTester']:	
		"""
		sample_l()

		Returns the radiance arriving at a
		given point, the incident direction, 
		pdf is used in MC sampling, i.e.,
		[Spectrum, wi, pdf, tester]
		"""
		raise NotImplementedError('src.core.light.{}.sample_l(): abstract method '
									'called'.format(self.__class__)) 		

	@abstractmethod
	def sample_r(self, scene: 'Scene', ls: 'LightSample', u1: FLOAT, 
					u2: FLOAT, time: FLOAT) -> ['geo.Ray', 'geo.Normal', FLOAT, 'Spectrum']:	
		"""
		sample_r()

		Samples a ray *leaving* the light source.

		Returns [geo.Ray, geo.Normal, pdf, spectrum]
		"""
		raise NotImplementedError('src.core.light.{}.sample_r(): abstract method '
									'called'.format(self.__class__)) 

	@abstractmethod
	def pdf(self, p: 'geo.Point', wi: 'geo.Vector') -> FLOAT:
		"""
		pdf()
		"""
		raise NotImplementedError('src.core.light.{}.pdf(): abstract method '
									'called'.format(self.__class__)) 		

	@property
	@abstractmethod
	def power(self, scene: 'Scene') -> 'Spectrum':
		"""
		power()

		Returns the total emitted power
		of the light.
		"""
		raise NotImplementedError('src.core.light.{}.power(): abstract method '
									'called'.format(self.__class__)) 	

	@abstractmethod
	def is_delta_light(self) -> bool:
		"""
		is_delta_light()

		Determines whether the light
		follows delta distribution, e.g.,
		point light, spotlight et.c.
		"""
		raise NotImplementedError('src.core.light.{}.is_delta_light(): abstract method '
									'called'.format(self.__class__)) 	

	def le(self, rd: 'geo.RayDifferential') -> 'Spectrum':
		"""
		le()

		Returns emitted radiance along
		a ray hit nothing
		"""
		return Spectrum(0.)


class PointLight(Light):
	"""
	PointLight Class
	
	By defauly positioned at the origin.
	"""
	def __init__(self, l2w: 'trans.Transform', intensity: 'Spectrum'):
		super().__init__(l2w)
		self.pos = l2w(geo.Point(0., 0., 0.)) # where the light is positioned in the world
		self.intensity = intensity

	def sample_l(self, p: 'geo.Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'geo.Vector', FLOAT, 'VisibilityTester']:	
		wi = geo.normalize(self.pos - p)
		pdf = 1.
		vis = VisibilityTester()
		vis.set_segment(p, pEps, self.pos, 0., time)
		return [self.intensity / (self.pos - p).sq_length(), wi, pdf, vis]

	def sample_r(self, scene: 'Scene', ls: 'LightSample', u1: FLOAT, 
					u2: FLOAT, time: FLOAT) -> ['geo.Ray', 'geo.Normal', FLOAT, 'Spectrum']:
		ray = geo.Ray(self.pos, mc.uniform_sample_sphere(ls.u_pos[0], ls.u_pos[1]),
						0., np.inf, time)
		Ns = geo.Normal.fromVector(ray.d)
		pdf = mc.uniform_sphere_pdf()
		return [ray, Ns, pdf, self.intensity]

	def pdf(self, p: 'geo.Point', wi: 'geo.Vector') -> FLOAT: return 0.

	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		return 4. * PI * self.intensity

	def is_delta_light(self) -> bool:
		return True


class SpotLight(Light):
	"""
	SpotLight Class
	
	By defauly positioned at the origin
	and light a cone towards +z.
	"""
	def __init__(self, l2w: 'trans.Transform', intensity: 'Spectrum', width: FLOAT, falloff: FLOAT):
		"""
		width: Overall angular width of the cone
		fall: angle at which falloff starts
		"""
		super().__init__(l2w)
		self.pos = l2w(geo.Point(0., 0., 0.)) # where the light is positioned in the world
		self.intensity = intensity
		self.cos_width = np.cos(np.deg2rad(width))
		self.cos_falloff = np.cos(np.deg2rad(falloff))

	def __falloff(self, w: 'geo.Vector') -> FLOAT:
		"""
		__falloff()

		Determines the falloff
		given a vector in the world.
		"""
		wl = geo.normalize(self.w2l(w))
		ct = wl.z
		if ct < self.cos_width:
			return 0.
		if ct > self.cos_falloff:
			return 1.
		# falloff inside the cone
		d = (ct - self.cos_width) / (self.cos_falloff - self.cos_width)
		return d * d * d * d

	def sample_l(self, p: 'geo.Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'geo.Vector', FLOAT, 'VisibilityTester']:	
		wi = geo.normalize(self.pos - p)
		pdf = 1.
		vis = VisibilityTester()
		vis.set_segment(p, pEps, self.pos, 0., time)
		return [self.intensity / self.__falloff(-wi), wi, pdf, vis]

	def sample_r(self, scene: 'Scene', ls: 'LightSample', u1: FLOAT, 
					u2: FLOAT, time: FLOAT) -> ['geo.Ray', 'geo.Normal', FLOAT, 'Spectrum']:
		
		v = mc.uniform_sample_cone(ls.u_pos[0], ls.u_pos[1], self.cos_width)

		ray = geo.Ray(self.pos, self.l2w(v), 0., np.inf, time)
		Ns = geo.Normal.fromVector(ray.d)
		pdf = mc.uniform_cone_pdf(self.cos_width)
		return [ray, Ns, pdf, self.intensity * self.__falloff(ray.d)]

	def pdf(self, p: 'geo.Point', wi: 'geo.Vector') -> FLOAT: return 0.

	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		"""
		power()

		approximate by the integral over
		spread angle cosine halfway
		between width and falloff
		"""
		return self.intensity * 2. * PI * (1. - .5 * (self.cos_falloff + self.cos_width))

	def is_delta_light(self) -> bool:
		return True


class ProjectionLight(Light):
	"""
	ProjectionLight Class
	
	Projecting light using texture.
	"""
	def __init__(self, l2w: 'trans.Transform', intensity: 'Spectrum', texname: str, fov: FLOAT):
		super().__init__(l2w)
		self.pos = l2w(geo.Point(0., 0., 0.)) # where the light is positioned in the world
		self.intensity = intensity

		# create a MIPMap
		try:
			import pytracer.utility.imageio as iio
			texels = iio.read_image(texname)
			width, height = np.shape(texels)
		except:
			print('src.core.texture.{}.get_texture(): cannot process file {}, '
					'use default one-valued MIPMap'.format(self.__class__, texname))
			texels = None
			width, height = 0, 0

		if texels is not None:
			from pytracer.texture import MIPMap
			from pytracer.spectral import RGBSpectrum
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
		self.proj_trans = trans.Transform.perspective(fov, self.hither, self.yon)

		# compute cosine of cone
		self.cos_width = np.cos(np.arctan(np.tan(np.deg2rad(fov) / 2.) * np.sqrt(1. + 1. / (aspect * aspect))))

	def __projection(self, w: 'geo.Vector') -> 'Spectrum':
		"""
		__projection()

		Utility method to determine
		the amount of light projected
		in the given direction.
		"""
		wl = self.w2l(w)
		# discard directions behind proj light
		if wl.z < self.hither:
			return Spectrum(0.)

		# project point onto plane
		pl = self.proj_trans(geo.Point(wl.x, wl.y, wl.z))
		if pl.x < self.scr_x0 or pl.x > self.scr_x1 or \
				pl.y < self.scr_y0 or pl.y > self.scr_y1:
			return Spectrum(0.)
		if self.projMap is None:
			return Spectrum(1.)

		s = (pl.x - self.scr_x0) / (self.scr_x1 - self.scr_x0)
		t = (pl.y - self.scr_y0) / (self.scr_y1 - self.scr_y0)
		return Spectrum(self.projMap.look_up([s, t]), SpectrumType.ILLUMINANT) 

	def sample_l(self, p: 'geo.Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'geo.Vector', FLOAT, 'VisibilityTester']:	
		wi = geo.normalize(self.pos - p)
		pdf = 1.
		vis = VisibilityTester()
		vis.set_segment(p, pEps, self.pos, 0., time)
		return [self.intensity / self.__projection(-wi), wi, pdf, vis]

	def sample_r(self, scene: 'Scene', ls: 'LightSample', u1: FLOAT, 
					u2: FLOAT, time: FLOAT) -> ['geo.Ray', 'geo.Normal', FLOAT, 'Spectrum']:
		
		v = mc.uniform_sample_cone(ls.u_pos[0], ls.u_pos[1], self.cos_width)

		ray = geo.Ray(self.pos, self.l2w(v), 0., np.inf, time)
		Ns = geo.Normal.fromVector(ray.d)
		pdf = mc.uniform_cone_pdf(self.cos_width)
		return [ray, Ns, pdf, self.intensity * self.__projection(ray.d)]

	def pdf(self, p: 'geo.Point', wi: 'geo.Vector') -> FLOAT: return 0.

	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		"""
		power()

		first order approximate by the 
		diagonal of the image, scaled
		by the average intensity.
		"""
		return Spectrum( self.projMap.look_up([.5, .5, .5]), SpectrumType.ILLUMINANT) * \
					self.intensity * 2. * PI * (1. - self.cos_width)

	def is_delta_light(self) -> bool:
		return True


class GonioPhotometricLight(Light):
	"""
	GonioPhotometricLight Class
	
	Projecting light using texture.
	"""
	def __init__(self, l2w: 'trans.Transform', intensity: 'Spectrum', texname: str):
		super().__init__(l2w)
		self.pos = l2w(geo.Point(0., 0., 0.)) # where the light is positioned in the world
		self.intensity = intensity

		# create a MIPMap
		try:
			import pytracer.utility.imageio as iio
			texels = iio.read_image(texname)
			width, height = np.shape(texels)
		except:
			print('src.core.texture.{}.get_texture(): cannot process file {}, '
					'use default one-valued MIPMap'.format(self.__class__, texname))
			texels = None		
		
		if texels is not None:
			from pytracer.texture import MIPMap
			from pytracer.spectral import RGBSpectrum
			self.MIPMap = MIPMap(RGBSpectrum, texels)
		else:
			self.MIPMap = None


	def __scale(self, w: 'geo.Vector') -> 'Spectrum':
		"""
		__scale()

		Utility method to scale
		the amount of light projected
		in the given direction. Assume
		the scale texture is encoded
		using spherical coordinates.
		"""
		if self.MIPMap is None:
			return Spectrum(1.)

		wp = geo.normalize(self.w2l(w))
		wp.z, wp.y = wp.y, wp.z
		theta = geo.spherical_theta(wp)
		phi = geo.spherical_phi(wp)
		s = phi * INV_2PI
		t = theta * INV_PI

		return Spectrum(self.projMap.look_up([s, t]), SpectrumType.ILLUMINANT)

	def sample_l(self, p: 'geo.Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'geo.Vector', FLOAT, 'VisibilityTester']:	
		wi = geo.normalize(self.pos - p)
		pdf = 1.
		vis = VisibilityTester()
		vis.set_segment(p, pEps, self.pos, 0., time)
		return [self.intensity / self.__scale(-wi), wi, pdf, vis]

	def sample_r(self, scene: 'Scene', ls: 'LightSample', u1: FLOAT, 
					u2: FLOAT, time: FLOAT) -> ['geo.Ray', 'geo.Normal', FLOAT, 'Spectrum']:
		ray = geo.Ray(self.pos, mc.uniform_sample_sphere(ls.u_pos[0], ls.u_pos[1]),
						0., np.inf, time)
		Ns = geo.Normal.fromVector(ray.d)
		pdf = mc.uniform_sphere_pdf()
		return [ray, Ns, pdf, self.intensity * self.__scale(ray.d)]

	def pdf(self, p: 'geo.Point', wi: 'geo.Vector') -> FLOAT: return 0.

	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		return 4. * PI * self.intensity * Spectrum(self.MIPMap.look_up([.5, .5, .5]), SpectrumType.ILLUMINANT)

	def is_delta_light(self) -> bool:
		return True


class DistantLight(Light):
	"""
	DistantLight Class
	
	Modelling distant or directional
	light, i.e., point light
	at infinity.
	"""
	def __init__(self, l2w: 'trans.Transform', radiance: 'Spectrum', di: 'geo.Vector'):
		super().__init__(l2w)
		self.di = geo.normalize(l2w(di))
		self.l = radiance

	def sample_l(self, p: 'geo.Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'geo.Vector', FLOAT, 'VisibilityTester']:	
		wi = self.di.copy()
		pdf = 1.
		vis = VisibilityTester()
		vis.set_ray(p, pEps, self.pos, wi, time)
		return [self.l.copy(), wi, pdf, vis]

	def sample_r(self, scene: 'Scene', ls: 'LightSample', u1: FLOAT, 
					u2: FLOAT, time: FLOAT) -> ['geo.Ray', 'geo.Normal', FLOAT, 'Spectrum']:
		"""
		Create a bounding disk
		and uniformly sample
		on it.
		"""
		# choose point on disk oriented towards light
		ctr, rad = scene.world_bound().bounding_sphere()
		_, v1, v2 = geo.coordinate_system(self.di)
		d1, d2 = mc.concentric_sample_disk(ls.u_pos[0], ls.u_pos[1])
		pnt = ctr + rad * (d1 * v1 + d2 * v2)

		# set ray
		ray = geo.Ray(pnt + rad * self.di, -self.di, 0., np.inf, time)
		Ns = geo.Normal.fromVector(ray.d)
		pdf = 1. / (PI * rad * rad)
		return [ray, Ns, pdf, self.l]

	def pdf(self, p: 'geo.Point', wi: 'geo.Vector') -> FLOAT: return 0.

	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		"""
		power()

		Approximate using a disk inside
		the scene's bounding sphere.
		"""
		_, rad = scene.world_bound().bounding_sphere()
	
		return self.l * PI * rad * rad	

	def is_delta_light(self) -> bool:
		return True


# Interface for Area Light
class AreaLight(Light):
	"""
	AreaLight Class
	"""
	def __init__(self, l2w: 'trans.Transform', ns: INT):
		super().__init__(l2w, ns)

	@abstractmethod
	def l(self, p: 'geo.Point', n: 'geo.Normal', w: 'geo.Vector') -> 'Spectrum':
		raise NotImplementedError('src.core.volume.{}.l(): abstract method '
									'called'.format(self.__class__)) 		


class DiffuseAreaLight(Light):
	"""
	DiffuseAreaLight Class
	
	Basic area light source with a
	uniform spatial and directional
	radiance distribution.
	"""
	def __init__(self, l2w: 'trans.Transform', emit: 'Spectrum', ns: INT, shape: 'Shape'):
		"""
		shape parameter will be a `ShapeSet`
		instance which subclasses `Shape`
		for easy implementation.
		"""
		super().__init__(l2w, ns)
		self.emit = emit
		self.shape_set = ShapeSet(shape)
		self.area = self.shape_set.areas

	# debugging purposes only
	def le(self, rd: 'geo.RayDifferential'):
		return self.emit

	def l(self, p: 'geo.Point', n: 'geo.Normal', w: 'geo.Vector') -> 'Spectrum':
		return self.emit if n.dot(w) > 0. else Spectrum(0.)

	# TODO use MCMC
	# def sample_l(self, p: 'geo.Point', pEps: FLOAT, ls: 'LightSample',
	# 		time: FLOAT,) -> ['Spectrum', 'geo.Vector', FLOAT, 'VisibilityTester']:	

	def sample_l(self, p: 'geo.Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'geo.Vector', FLOAT, 'VisibilityTester']:	
		ps, ns = self.shape_set.sample_p(p, ls)
		wi = geo.normalize(ps - p)
		pdf = self.shape_set.pdf(p, wi)
		vis = VisibilityTester()
		vis.set_segment(p, pEps, ps, EPS, time)
		return [self.l(ps, ns, -wi), wi, pdf, vis]

	def sample_r(self, scene: 'Scene', ls: 'LightSample', u1: FLOAT, 
					u2: FLOAT, time: FLOAT) -> ['geo.Ray', 'geo.Normal', FLOAT, 'Spectrum']:
		"""
		Create a bounding disk
		and uniformly sample
		on it.
		"""
		org, Ns = self.shape_set.sample(ls)
		di = mc.uniform_sample_sphere(u1, u2)
		if di.dot(Ns) < 0.:
			di *= -1.
		ray = geo.Ray(org, di, EPS, np.inf, time)
		pdf = self.shape_set.pdf(org) * INV_2PI
		return [ray, Ns, pdf, self.l(org, Ns, di)]

	def pdf(self, p: 'geo.Point', wi: 'geo.Vector') -> FLOAT: return self.shape_set.pdf(p, wi)



	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		return self.emit * self.area * PI

	def is_delta_light(self) -> bool:
		return False


class InfiniteAreaLight(Light):
	"""
	InfiniteAreaLight Class
	
	Models environment lighting
	"""
	def __init__(self, l2w: 'trans.Transform', l: 'Spectrum', ns: INT, texmap: str=None):
		"""
		width: Overall angular width of the cone
		fall: angle at which falloff starts
		"""
		super().__init__(l2w, ns)
		self.pos = l2w(geo.Point(0., 0., 0.)) # where the light is positioned in the world
		texels = np.array([[l.to_rgb()]])
		width, height = 1, 1

		if texmap is not None:
			try:
				import pytracer.utility.imageio as iio
				texels = iio.read_image(texmap)
				width, height = np.shape(texels)
			except:
				print('src.core.texture.{}.get_texture(): cannot process file {}, '
				'use default one-valued MIPMap'.format(self.__class__, texmap))
				texels = np.array([[l.to_rgb()]])
				width, height = 1, 1

		from pytracer.texture import MIPMap
		from pytracer.spectral import RGBSpectrum
		self.radMap = MIPMap(RGBSpectrum, texels)


		# init sampling PDFs <725>
		# compute image for envir. map
		img = np.empty([height, width], dtype=FLOAT)
		filt = 1. / max(width, height)

		for v in range(height):
			vp = v / height
			st = np.sin(PI * (v + .5) / height)
			for u in range(width):
				up = u / width
				img[v][u] = self.radMap.look_up([up, vp, filt]).y()

		# compute sampling distribution
		self.dist = mc.Distribution2D(img)

	def le(self, rd: 'geo.RayDifferential') -> 'Spectrum':
		wh = geo.normalize(self.w2l(rd.d))
		s = geo.spherical_phi(wh) * INV_2PI
		t = geo.spherical_theta(wh) * INV_PI
		return Spectrum.from_rgb(self.radMap.look_up(s, t), SpectrumType.ILLUMINANT)

	def sample_l(self, p: 'geo.Point', pEps: FLOAT, ls: 'LightSample',
			time: FLOAT,) -> ['Spectrum', 'geo.Vector', FLOAT, 'VisibilityTester']:	
		# find (u, v) sample coords in inf. light texture
		uv, pdf = self.dist.sample_cont(ls.u_pos[0], ls.u_pos[1])

		# convert inf light sample pnt to direction
		theta = uv[1] * PI
		phi = uv[0] * 2. * PI
		ct = np.cos(theta)
		st = np.sin(theta)
		sp = np.sin(phi)
		cp = np.cos(phi)

		wi = self.l2w(geo.Vector(st * cp, st * sp, ct))

		# compute pdf for sampled inf light direction
		if st == 0.:
			pdf = 0.
		pdf = pdf / (2. * PI * PI * st)


		# return radiance value
		vis = VisibilityTester()
		vis.set_ray(p, pEps, wi, time)

		return [Spectrum.from_rgb(self.radMap.look_up([uv[0], uv[1]]), SpectrumType.ILLUMINANT),
					wi, pdf, vis]

	def pdf(self, p: 'geo.Point', w: 'geo.Vector') -> FLOAT:
		wi = self.w2l(w)
		theta = geo.spherical_theta(wi)
		phi = geo.spherical_phi(wi)
		st = np.sin(theta)
		if st == 0.:
			return 0.
		return self.dist.pdf(phi * INV_2PI, theta * INV_PI / 
				(2. * PI * PI * st))

	def sample_r(self, scene: 'Scene', ls: 'LightSample', u1: FLOAT, 
					u2: FLOAT, time: FLOAT) -> ['geo.Ray', 'geo.Normal', FLOAT, 'Spectrum']:
		"""
		Create a bounding disk
		and uniformly sample
		on it.
		"""
		# find (u, v) sample coords in inf. light texture
		uv, pdf = self.dist.sample_cont(ls.u_pos[0], ls.u_pos[1])
		if pdf == 0.:
			return [None, None, 0., Spectrum(0.)]

		theta = uv[1] * PI
		phi = uv[0] * 2. * PI
		ct = np.cos(theta)
		st = np.sin(theta)
		sp = np.sin(phi)
		cp = np.cos(phi)
		d = -self.l2w(geo.Vector(st * cp, st * sp, ct))
		Ns = geo.Normal.fromVector(d)

		# choose point on disk oriented towards light
		ctr, rad = scene.world_bound().bounding_sphere()
		_, v1, v2 = geo.coordinate_system(self.di)
		d1, d2 = mc.concentric_sample_disk(ls.u_pos[0], ls.u_pos[1])
		pnt = ctr + rad * (d1 * v1 + d2 * v2)

		# set ray
		ray = geo.Ray(pnt + rad * (-d), d, 0., np.inf, time)

		# compute pdf
		dir_pdf = pdf / (2. * PI * PI * st)
		area_pdf = 1. / (PI * rad * rad)
		pdf = dir_pdf * area_pdf
		if st == 0.:
			pdf == 0.

		return [ray, Ns, pdf, Spectrum.from_rgb(self.radMap.look_up([uv[0], uv[1]]), SpectrumType.ILLUMINANT)]

	@property
	def power(self, scene: 'Scene') -> 'Spectrum':
		_, rad = scene.world_bound().bounding_sphere()
		return PI * rad * rad * Spectrum.from_rgb(self.radMap.look_up([.5, .5, .5]), SpectrumType.ILLUMINANT)

	def is_delta_light(self) -> bool:
		return False






















