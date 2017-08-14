"""
disk.py

Disk implementation.

Created by Jiayao on July 27, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo
import pytracer.transform as trans
from pytracer.shape import Shape

__all__ = ['Disk']

class Disk(Shape):
	"""
	Disk Class

	Subclasses `Shape` and is used
	to model possibly partial Disk.
	"""

	def __init__(self, o2w: 'trans.Transform', w2o: 'trans.Transform',
	             ro: bool, ht: FLOAT, r: FLOAT, ri: FLOAT, pm: FLOAT):
		super().__init__(o2w, w2o, ro)
		self.height = ht
		self.radius = r
		self.inner_radius = ri
		self.phiMax = np.deg2rad(np.clip(pm, 0., 360.))

	def __repr__(self):
		return "{}\nRadius: {}\nInner Radius: {}" \
			.format(self.__class__, self.radius, self.inner_radius)

	def object_bound(self) -> 'geo.BBox':
		return geo.BBox(geo.Point(-self.radius, -self.radius, self.height),
		            geo.Point(self.radius, self.radius, self.height))

	def sample(self, u1: FLOAT, u2: FLOAT) -> ['geo.Point', 'geo.Normal']:

		# account for partial disk
		from ..montecarlo import concentric_sample_disk
		x, y = concentric_sample_disk(u1, u2)
		phi = np.arctan2(y, x) * self.phiMax * INV_2PI
		r = self.inner + np.sqrt(x * x + y * y) * (self.radius - self.inner)

		p = geo.Point(r * np.cos(phi), r * np.sin(phi), self.height)

		Ns = geo.normalize(self.o2w(p))
		if self.ro:
			Ns *= -1.

		return [self.o2w(p), Ns]

	def intersect(self, r: 'geo.Ray') -> (bool, FLOAT, FLOAT, 'geo.DifferentialGeometry'):
		"""
		Returns:
		 - bool: specify whether intersects
		 - tHit: hit point param
		 - rEps: error tolerance
		 - dg: geo.DifferentialGeometry object
		"""
		# transform ray to object space
		ray = self.w2o(r)

		# parallel ray has not intersection
		if np.fabs(ray.d.z) < EPS:
			return [False, None, None, None]

		thit = (self.height - ray.o.z) / ray.d.z
		if thit < ray.mint or thit > ray.maxt:
			return [False, None, None, None]

		phit = ray(thit)

		# check radial distance
		dt2 = phit.x * phit.x + phit.y * phit.y
		if dt2 > self.radius * self.radius or \
						dt2 < self.inner_radius * self.inner_radius:
			return [False, None, None, None]

		# check angle
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0:
			phi += 2. * np.pi

		if phi > self.phiMax:
			return [False, None, None, None]

		# otherwise ray hits the disk
		# initialize the differential structure
		u = phi / self.phiMax
		v = 1. - ((np.sqrt(dt2 - self.inner_radius)) /
		          (self.radius - self.inner_radius))

		# find derivatives
		dpdu = geo.Vector(-self.phiMax * phit.y, self.phiMax * phit.x, 0.)
		dpdv = geo.Vector(-phit.x / (1. - v), -phit.y / (1. - v), 0.)
		dpdu *= self.phiMax * INV_2PI
		dpdv *= (self.radius - self.inner_radius) / self.radius

		# derivative of geo.Normals
		dndu = geo.Normal(0., 0., 0., )
		dndv = geo.Normal(0., 0., 0., )

		o2w = self.o2w
		dg = geo.DifferentialGeometry(o2w(phit), o2w(dpdu), o2w(dpdv),
		                          o2w(dndu), o2w(dndv), u, v, self)
		return True, thit, EPS * thit, dg

	def intersect_p(self, r: 'geo.Ray') -> bool:
		# transform ray to object space
		ray = self.w2o(r)

		# parallel ray has not intersection
		if np.fabs(ray.d.z) < EPS:
			return False

		thit = (self.height - ray.o.z) / ray.d.z
		if thit < ray.mint or thit > ray.maxt:
			return False

		phit = ray(thit)

		# check radial distance
		dt2 = phit.x * phit.x + phit.y * phit.y
		if dt2 > self.radius * self.radius or \
						dt2 < self.inner_radius * self.inner_radius:
			return False

		# check angle
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0:
			phi += 2. * np.pi

		if phi > self.phiMax:
			return False

		return True

	def area(self) -> FLOAT:
		return self.phiMax * .5 * \
		       (self.radius * self.radius - self.inner_radius * self.inner_radius)
