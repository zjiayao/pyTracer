"""
cylinder.py

Cylinder implementation.

Created by Jiayao on July 27, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from .. import *
from . import Shape
from .. import geometry as geo
from .. import transform as trans

__all__ = ['Cylinder']

class Cylinder(Shape):
	"""
	Cylinder Class

	Subclasses `Shape` and is used
	to model possibly partial Cylinder.
	"""

	def __init__(self, o2w: 'trans.Transform', w2o: 'trans.Transform',
	             ro: bool, rad: FLOAT, z0: FLOAT, z1: FLOAT, pm: FLOAT):
		super().__init__(o2w, w2o, ro)
		self.radius = rad
		self.zmin = min(z0, z1)
		self.zmax = max(z0, z1)
		self.phiMax = np.deg2rad(np.clip(pm, 0., 360.))

	def __repr__(self):
		return "{}\nRadius: {}".format(self.__class__, self.radius)

	def object_bound(self) -> 'geo.BBox':
		return geo.BBox(geo.Point(-self.radius, -self.radius, self.zmin),
		            geo.Point(self.radius, self.radius, self.zmax))

	def sample(self, u1: FLOAT, u2: FLOAT) -> ['geo.Point', 'geo.Normal']:
		z = util.lerp(u1, self.zmin, self.zmax)
		t = u2 * self.phiMax
		p = geo.Point(self.radius * np.cos(t), self.radius * np.sin(t), z)
		Ns = geo.normalize(self.o2w(geo.Normal(p.x, p.y, 0.)))

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

		# solve quad eqn
		A = ray.d.x * ray.d.x + ray.d.y * ray.d.y
		B = 2 * (ray.d.x * ray.o.x + ray.d.y * ray.o.y)
		C = ray.o.x * ray.o.x + ray.o.y * ray.o.y - self.radius * self.radius

		D = B * B - 4. * A * C
		if D <= 0.:
			return [False, None, None, None]

		# validate solutions
		[t0, t1] = np.roots([A, B, C])
		if t0 > t1:
			t0, t1 = t1, t0
		if t0 > ray.maxt or t1 < ray.mint:
			return [False, None, None, None]
		thit = t0
		if t0 < ray.mint:
			thit = t1
			if thit > ray.maxt:
				return [False, None, None, None]

		# cylinder hit position
		phit = ray(thit)
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0.:
			phi += 2. * np.pi

		# test intersection against clipping params
		if phit.z < self.zmin or phit.z > self.zmax or phi > self.phiMax:
			if thit == t1 or t1 > ray.maxt:
				return [False, None, None, None]

			# try again with t1
			thit = t1
			phit = ray(thit)
			phi = np.arctan2(phit.y, phit.x)
			if phi < 0.:
				phi += 2. * np.pi

			if phit.z < self.zmin or phit.z > self.zmax or phi > self.phiMax:
				if thit == t1 or t1 > ray.maxt:
					return [False, None, None, None]

		# otherwise ray hits the cylinder
		# initialize the differential structure
		u = phi / self.phiMax
		v = (phit.z - self.zmin) / (self.thetaMax - self.thetaMin)

		# find derivatives
		dpdu = geo.Vector(-self.phiMax * phit.y, self.phiMax * phit.x, 0.)
		dpdv = geo.Vector(0., 0., self.zmax - self.zmin)

		# derivative of geo.Normals
		# given by Weingarten Eqn
		d2pduu = -self.phiMax * self.phiMax * geo.Vector(phit.x, phit.y, 0.)
		d2pduv = geo.Vector(0., 0., 0.)
		d2pdvv = geo.Vector(0., 0., 0.)

		# fundamental forms
		E = dpdu.dot(dpdu)
		F = dpdu.dot(dpdv)
		G = dpdv.dot(dpdv)
		N = geo.normalize(dpdu.cross(dpdv))
		e = N.dot(d2pduu)
		f = N.dot(d2pduv)
		g = N.dot(d2pdvv)

		invEGFF = 1. / (E * G - F * F)
		dndu = geo.Normal.from_arr((f * F - e * G) * invEGFF * dpdu +
		                         (e * F - f * E) * invEGFF * dpdv)

		dndv = geo.Normal.from_arr((g * F - f * G) * invEGFF * dpdu +
		                         (f * F - g * E) * invEGFF * dpdv)

		o2w = self.o2w
		dg = geo.DifferentialGeometry(o2w(phit), o2w(dpdu), o2w(dpdv),
		                          o2w(dndu), o2w(dndv), u, v, self)
		return True, thit, EPS * thit, dg

	def intersect_p(self, r: 'geo.Ray') -> bool:
		# transform ray to object space
		ray = self.w2o(r)

		# solve quad eqn
		A = ray.d.x * ray.d.x + ray.d.y * ray.d.y
		B = 2 * (ray.d.x * ray.o.x + ray.d.y * ray.o.y)
		C = ray.o.x * ray.o.x + ray.o.y * ray.o.y - self.radius * self.radius

		D = B * B - 4. * A * C
		if D <= 0.:
			return False

		# validate solutions
		[t0, t1] = np.roots([A, B, C])
		if t0 > t1:
			t0, t1 = t1, t0
		if t0 > ray.maxt or t1 < ray.mint:
			return False
		thit = t0
		if t0 < ray.mint:
			thit = t1
			if thit > ray.maxt:
				return False

		# cylinder hit position
		phit = ray(thit)
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0.:
			phi += 2. * np.pi

		# test intersection against clipping params
		if phit.z < self.zmin or phit.z > self.zmax or phi > self.phiMax:
			if thit == t1 or t1 > ray.maxt:
				return False

			# try again with t1
			thit = t1
			phit = ray(thit)
			phi = np.arctan2(phit.y, phit.x)
			if phi < 0.:
				phi += 2. * np.pi

			if phit.z < self.zmin or phit.z > self.zmax or phi > self.phiMax:
				if thit == t1 or t1 > ray.maxt:
					return False

		return True

	def area(self) -> FLOAT:
		return self.phiMax * self.radius * (self.zmax - self.zmin)
