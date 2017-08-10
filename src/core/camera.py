'''
camera.py

The base class to model cameras.

Created by Jiayao on July 31, 2017
'''

from numba import jit
from abc import ABCMeta, abstractmethod  
import numpy as np
from src.core.pytracer import *


class Camera(object, metaclass=ABCMeta):
	'''
	Camera Class
	'''

	def __init__(self, c2w: 'AnimatedTransform', s_open: FLOAT,
					s_close: FLOAT, film: 'Film'):
		self.c2w = c2w
		self.s_open = s_open
		self.s_close = s_close
		self.film = film

	def __repr__(self):
		return "{}\nShutter: {} - {}".format(self.__class__, self.s_open, self.s_close)

	@abstractmethod
	def generate_ray(self, sample: 'CameraSample') -> [FLOAT, 'Ray']:
		'''
		Generate ray based on image sample.
		Returned ray direction is normalized

		@param
			- sample: instance of `CameraSample` class
		@return
			- FLOAT: light weight
			- Ray: generated `Ray` object
		'''
		raise NotImplementedError('src.core.camera {}.generate_ray: abstract method called' \
									.format(self.__class__)) 

	def generate_ray_differential(self, sample: 'CameraSample') -> [FLOAT, 'RayDifferential']:
		'''
		Generate ray differential.
		'''
		wt, rd = self.generate_ray(sample)
		rd = RayDifferential.fromRay(rd)
		
		# find ray shift along x
		xshift = CameraSample.fromSample(sample)
		xshift.imageX += 1
		wtx, rx = self.generate_ray(xshift)
		rd.rxOrigin = rx.o
		rd.rxDirection = rx.d

		# find ray shift along y
		yshift = CameraSample.fromSample(sample)
		yshift.imageY += 1
		wty, ry = self.generate_ray(yshift)
		rd.ryOrigin = ry.o
		rd.ryDirection = ry.d

		if wtx == 0. or wty == 0.:
			return [0., rd]

		rd.hasDifferentials = True
		return [wt, rd]

class ProjectiveCamera(Camera):
	'''
	ProjectiveCamera Class
	'''
	def __init__(self, c2w: 'AnimatedTransform', proj: 'Transform', scr_win: [FLOAT],
					s_open: FLOAT, s_close: FLOAT, lensr: FLOAT, focald: FLOAT, f: 'Film'):
		super().__init__(c2w, s_open, s_close, f)

		# set dof prarms
		self.lens_rad = lensr
		self.focal_dist = focald		

		# compute transfomations
		self.c2s = proj
		## compute projective screen transfomations
		s2r = Transform.scale(f.xResolution, f.yResolution, 1.) * \
				Transform.scale(1./(scr_win[1] - scr_win[0]),
								1./(scr_win[2] - scr_win[3]), 1.) * \
				Transform.translate(Vector(-scr_win[0], -scr_win[3], 0.)) # upper-left corner to origin

		r2s = s2r.inverse()
		self.r2c = self.c2s.inverse() * r2s


class OrthoCamera(ProjectiveCamera):
	'''
	OrthoCamera

	Models orthographic camera
	'''
	def __init__(self, c2w: 'AnimatedTransform', scr_win: [FLOAT],
					s_open: FLOAT, s_close: FLOAT, lensr: FLOAT, focald: FLOAT, f: 'Film'):
		super().__init__(c2w, Transform.orthographic(0., 1.), scr_win, s_open,
							s_close, lensr, focald, f)
		# compute differential changes in origin
		self.dxCam = self.r2c(Vector(1., 0., 0.))
		self.dyCam = self.r2c(Vector(0., 1., 0.))


	def generate_ray(self, sample: 'CameraSample') -> [FLOAT, 'Ray']:
		'''
		Generate ray based on image sample.
		Returned ray direction is normalized
		'''
		# generate raster and camera samples
		Pras = Point(sample.imageX, sample.imageY, 0.)
		Pcam = self.r2c(Pras)

		ray = Ray(Pcam, Vector(0., 0., 1.), 0., np.inf)
		
		# modify ray for dof
		if lensr > 0.:
			# sample point on lens
			## todo: concentric_sample_disk
			lens_u, lens_v = \
				src.sampler.monte_carlo.concentric_sample_disk(sample.lens_u, sample.lens_v)
			lens_u *= self.lens_rad
			lens_v *= self.lens_rad

			# compute point on focal plane
			ft = self.focal_dist / ray.d.z
			Pfoc = ray(ft)

			# update ray
			ray.o = Point(lens_u, lens_v, 0.)
			ray.d = normalize(Pfoc - ray.o)
		
		ray.time = Lerp(sample.time, self.s_open, self.s_close)
		ray = self.c2w(ray)
		return [1., ray]

	def generate_ray_differential(self, sample: 'CameraSample') -> [FLOAT, 'RayDifferential']:
		'''
		Generate ray differential.
		'''
		wt, rd = self.generate_ray(sample)
		rd = RayDifferential.fromRay(rd)
		
		# find ray shift along x
		rd.rxOrigin = r.o + self.dxCam
		rd.ryOrigin = r.o + self.dyCam
		rd.rxDirection = rd.ryDirection = rd.d
		rd.hasDifferentials = True
		rd = self.c2w(rd)

		return [wt, rd]		

class PerspectiveCamera(ProjectiveCamera):
	'''
	PerspectiveCamera

	Models perspective camera
	'''
	def __init__(self, c2w: 'AnimatedTransform', scr_win: [FLOAT],
					s_open: FLOAT, s_close: FLOAT, lensr: FLOAT, focald: FLOAT, fov: FLOAT, f: 'Film'):
		super().__init__(c2w, Transform.perspective(fov, .001, 1000.), # non-raster based, set arbitrarily
				scr_win, s_open, s_close, lensr, focald, f)
		# compute differential changes in origin
		self.dxCam = self.r2c(Point(1., 0., 0.)) - self.r2c(Point(0., 0., 0.))
		self.dyCam = self.r2c(Point(0., 1., 0.)) - self.r2c(Point(0., 0., 0.))


	def generate_ray(self, sample: 'CameraSample') -> [FLOAT, 'Ray']:
		'''
		Generate ray based on image sample.
		Returned ray direction is normalized
		'''
		# generate raster and camera samples
		Pras = Point(sample.imageX, sample.imageY, 0.)
		Pcam = self.r2c(Pras)

		ray = Ray(Point(0., 0., 0.), Pcam, 0., np.inf) # ray.d is a Vector init from a Point
		# modify ray for dof
		if lensr > 0.:
			# sample point on lens
			## todo: concentric_sample_disk
			lens_u, lens_v = \
				src.sampler.monte_carlo.concentric_sample_disk(sample.lens_u, sample.lens_v)
			lens_u *= self.lens_rad
			lens_v *= self.lens_rad

			# compute point on focal plane
			ft = self.focal_dist / ray.d.z
			Pfoc = ray(ft)

			# update ray
			ray.o = Point(lens_u, lens_v, 0.)
			ray.d = normalize(Pfoc - ray.o)

		ray.time = Lerp(sample.time, self.s_open, self.s_close)
		ray = self.c2w(ray)
		return [1., ray]

	def generate_ray_differential(self, sample: 'CameraSample') -> [FLOAT, 'RayDifferential']:
		'''
		Generate ray differential.
		'''
		wt, rd = self.generate_ray(sample)
		rd = RayDifferential.fromRay(rd)
		
		# find ray shift along x
		rd.rxOrigin = rd.ryOrigin = r.o
		rd.rxDirection = normalize(self.dxCam + Pcam) # Vector + Point => Vector
		rd.ryDirection = normalize(self.dyCam + Pcam)
		rd.hasDifferentials = True
		rd = self.c2w(rd)
		
		return [wt, rd]	

class EnvironmentCamera(Camera):
	'''
	EnvironmentCamera

	Models equirectangular projection of the scene
	'''
	def __init__(self, c2w: 'AnimatedTransform', s_open: FLOAT, s_close: FLOAT, f: 'Film'):
		super().__init__(c2w, s_open, s_close, f)

	def generate_ray(self, sample: 'CameraSample') -> [FLOAT, 'Ray']:
		'''
		Generate ray based on image sample.
		Returned ray direction is normalized
		'''
		time = Lerp(sample.time, self.s_open, self.s_close)

		# compute ray direction
		theta = np.pi * sample.imageY / f.yResolution
		phi = 2 * np.pi * sample.imageX / f.xResolution
		stheta = np.sin(theta)

		ray = self.c2w( Ray(Point(0.,0.,0.), 
			Vector(stheta * np.cos(phi), np.cos(theta), stheta * np.sin(phi)), 0., np.inf, time) )
		return [1., ray]

