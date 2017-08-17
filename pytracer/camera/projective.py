"""
projective.py

Implements projective camera
and variants, includes:
	- Orthographic Camera
	- Perspective Camera

Created by Jiayao on July 31, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo
import pytracer.transform as trans
from pytracer.camera.camera import Camera
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from pytracer.sampler import CameraSample, StratifiedSampler

__all__ = ['ProjectiveCamera', 'OrthoCamera', 'PerspectiveCamera', 'PinholeCamera']


class PinholeCamera(Camera):
	def __init__(self, c2w: 'trans.AnimatedTransform', focald: FLOAT, f: 'Film'):
		super().__init__(c2w, 0., 0., f)
		self.focal = focald

	def generate_ray(self, sample: 'CameraSample') -> [FLOAT, 'geo.Ray']:
		p = geo.Point(sample.imageX, sample.imageY, 0.)
		pp = geo.Point(0., 0., self.focal)
		d = self.c2w.startTransform(pp - p)
		p = self.c2w.startTransform(p)
		return 1., geo.Ray(p, d)




class ProjectiveCamera(Camera):
	"""
	ProjectiveCamera Class
	"""

	def __init__(self, c2w: 'trans.AnimatedTransform', proj: 'trans.Transform', scr_win: [FLOAT],
				 s_open: FLOAT, s_close: FLOAT, lensr: FLOAT, focald: FLOAT, f: 'Film'):
		super().__init__(c2w, s_open, s_close, f)

		# set dof prarms
		self.lens_rad = lensr
		self.focal_dist = focald

		# compute transfomations
		self.c2s = proj
		## compute projective screen transfomations
		s2r = trans.Transform.scale(f.xResolution, f.yResolution, 1.) * \
			  trans.Transform.scale(1. / (scr_win[1] - scr_win[0]),
							  1. / (scr_win[2] - scr_win[3]), 1.) * \
			  trans.Transform.translate(geo.Vector(-scr_win[0], -scr_win[3], 0.))  # upper-left corner to origin

		r2s = s2r.inverse()
		self.r2c = self.c2s.inverse() * r2s

	def generate_ray(self, sample: 'CameraSample') -> [FLOAT, 'geo.Ray']:
		pass


class OrthoCamera(ProjectiveCamera):
	"""
	OrthoCamera

	Models orthographic camera
	"""

	def __init__(self, c2w: 'trans.AnimatedTransform', scr_win: [FLOAT],
				 s_open: FLOAT, s_close: FLOAT, lensr: FLOAT, focald: FLOAT, f: 'Film'):
		super().__init__(c2w, trans.Transform.orthographic(0., 1.), scr_win, s_open,
						 s_close, lensr, focald, f)
		# compute differential changes in origin
		self.dxCam = self.r2c(geo.Vector(1., 0., 0.))
		self.dyCam = self.r2c(geo.Vector(0., 1., 0.))

	def generate_ray(self, sample: 'CameraSample') -> [FLOAT, 'geo.Ray']:
		"""
		Generate ray based on image sample.
		Returned ray direction is normalized
		"""
		# generate raster and camera samples
		Pras = geo.Point(sample.imageX, sample.imageY, 0.)
		Pcam = self.r2c(Pras)

		ray = geo.Ray(Pcam, geo.Vector(0., 0., 1.), 0., np.inf)

		# modify ray for dof
		if self.lens_rad > 0.:
			# sample point on lens

			from pytracer.montecarlo import concentric_sample_disk
			lens_u, lens_v = \
				concentric_sample_disk(sample.lens_u, sample.lens_v)
			lens_u *= self.lens_rad
			lens_v *= self.lens_rad

			# compute point on focal plane
			ft = self.focal_dist / ray.d.z
			Pfoc = ray(ft)

			# update ray
			ray.o = geo.Point(lens_u, lens_v, 0.)
			ray.d = geo.normalize(Pfoc - ray.o)

		ray.time = util.lerp(sample.time, self.s_open, self.s_close)
		ray = self.c2w(ray)
		return [1., ray]

	def generate_ray_differential(self, sample: 'CameraSample') -> [FLOAT, 'geo.RayDifferential']:
		"""
		Generate ray differential.
		"""
		wt, rd = self.generate_ray(sample)
		rd = geo.RayDifferential.from_ray(rd)

		# find ray shift along x
		rd.rxOrigin = rd.o + self.dxCam
		rd.ryOrigin = rd.o + self.dyCam
		rd.rxDirection = rd.ryDirection = rd.d
		rd.hasDifferentials = True
		rd = self.c2w(rd)

		return [wt, rd]


class PerspectiveCamera(ProjectiveCamera):
	"""
	PerspectiveCamera

	Models perspective camera
	"""

	def __init__(self, c2w: 'trans.AnimatedTransform', scr_win: [FLOAT],
				 s_open: FLOAT, s_close: FLOAT, lensr: FLOAT, focald: FLOAT, fov: FLOAT, f: 'Film'):
		super().__init__(c2w, trans.Transform.perspective(fov, .001, 1000.),  # non-raster based, set arbitrarily
						 scr_win, s_open, s_close, lensr, focald, f)
		# compute differential changes in origin
		self.dxCam = self.r2c(geo.Point(1., 0., 0.)) - self.r2c(geo.Point(0., 0., 0.))
		self.dyCam = self.r2c(geo.Point(0., 1., 0.)) - self.r2c(geo.Point(0., 0., 0.))

	def generate_ray(self, sample: 'CameraSample') -> [FLOAT, 'geo.Ray']:
		"""
		Generate ray based on image sample.
		Returned ray direction is normalized
		"""
		# generate raster and camera samples
		Pras = geo.Point(sample.imageX, sample.imageY, 0.)
		Pcam = self.r2c(Pras)

		ray = geo.Ray(geo.Point(0., 0., 0.), geo.Vector.from_arr(geo.normalize(Pcam)), 0., np.inf)  # ray.d is a geo.Vector init from a geo.Point
		# modify ray for dof
		if self.lens_rad > 0.:
			# sample point on lens

			from pytracer.montecarlo import concentric_sample_disk

			lens_u, lens_v = \
				concentric_sample_disk(sample.lens_u, sample.lens_v)
			lens_u *= self.lens_rad
			lens_v *= self.lens_rad

			# compute point on focal plane
			ft = self.focal_dist / ray.d.z
			Pfoc = ray(ft)

			# update ray
			ray.o = geo.Point(lens_u, lens_v, 0.)
			ray.d = geo.normalize(Pfoc - ray.o)

		ray.time = util.lerp(sample.time, self.s_open, self.s_close)
		ray = self.c2w(ray)
		return [1., ray]

	def generate_ray_differential(self, sample: 'CameraSample') -> [FLOAT, 'geo.RayDifferential']:
		"""
		Generate ray differential.
		"""
		p_ras = geo.Point(sample.imageX, sample.imageY, 0.)
		p_cam = self.r2c(p_ras)

		ray = geo.RayDifferential(geo.Point(0., 0., 0.), geo.Vector.from_arr(geo.normalize(p_cam)), 0., np.inf)  # ray.d is a geo.Vector init from a geo.Point

		from pytracer.montecarlo import concentric_sample_disk
		if self.lens_rad > 0.:
			# depth of field

			lens_u, lens_v = \
				concentric_sample_disk(sample.lens_u, sample.lens_v)
			lens_u *= self.lens_rad
			lens_v *= self.lens_rad

			# compute point on focal plane
			ft = self.focal_dist / ray.d.z
			Pfoc = ray(ft)

			# update ray
			ray.o = geo.Point(lens_u, lens_v, 0.)
			ray.d = geo.normalize(Pfoc - ray.o)

		if self.lens_rad > 0.:
			# with defocus blue
			lens_u, lens_v = concentric_sample_disk(sample.lens_u, sample.lens_v)
			lens_u *= self.lens_rad
			lens_v *= self.lens_rad

			# compute point on focal plane
			dx = geo.normalize(self.dxCam + p_cam)
			ft = self.focal_dist / dx.z
			Pfoc = geo.Point(0., 0., 0.) + ft * dx
			ray.rxOrigin = geo.Point(lens_u, lens_v, 0.)
			ray.rxDirection = geo.normalize(Pfoc - ray.rxOrigin)

			dy = geo.normalize(geo.Vector.from_arr(p_cam + self.dyCam))
			ft = self.focal_dist / dy.z
			Pfoc = geo.Point(0., 0., 0.) + ft * dy
			ray.ryOrigin = geo.Point(lens_u, lens_v, 0.)
			ray.ryDirection = geo.normalize(Pfoc - ray.ryOrigin)

		else:
			ray.rxOrigin = ray.ryOrigin = ray.o
			ray.rxDirection = geo.normalize(self.dxCam + p_cam)  # geo.Vector + geo.Point => geo.Vector
			ray.ryDirection = geo.normalize(self.dyCam + p_cam)

		ray.time = sample.time
		ray = self.c2w(ray)
		ray.hasDifferentials = True

		return [1., ray]

