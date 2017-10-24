"""
sphere.py

Sphere implementation.

Created by Jiayao on July 27, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo
import pytracer.transform as trans
from pytracer.shape import Shape

__all__ = ['Sphere']


class Sphere(Shape):
	"""
	Sphere Class

	Subclasses `Shape` and is used
	to model possibly partial Sphere.
	"""

	def __init__(self, o2w: 'trans.Transform', w2o: 'trans.Transform',
	             ro: bool, rad: FLOAT, z0: FLOAT, z1: FLOAT, pm: FLOAT):
		super().__init__(o2w, w2o, ro)
		self.radius = rad
		self.zmin = np.clip(min(z0, z1), -rad, rad)
		self.zmax = np.clip(max(z0, z1), -rad, rad)
		self.thetaMin = np.arccos(np.clip(self.zmin / rad, -1., 1.))
		self.thetaMax = np.arccos(np.clip(self.zmax / rad, -1., 1.))
		self.phiMax = np.deg2rad(np.clip(pm, 0., 360.))

	def __repr__(self):
		return "{}\nRadius: {}".format(self.__class__, self.radius)

	def object_bound(self) -> 'geo.BBox':
		return geo.BBox(geo.Point(-self.radius, -self.radius, self.zmin),
		            geo.Point(self.radius, self.radius, self.zmax))

	def sample(self, u1: FLOAT, u2: FLOAT) -> ['geo.Point', 'geo.Normal']:
		"""
		account for partial sphere
		"""
		from pytracer.montecarlo import uniform_sample_sphere
		v = uniform_sample_sphere(u1, u2)

		phi = geo.spherical_theta(v) * self.phiMax * INV_2PI
		theta = self.thetaMin + geo.spherical_theta(v) * (self.thetaMax - self.thetaMin)

		v = geo.spherical_direction(np.sin(theta), np.cos(theta), phi) * self.radius
		v.z = self.zmin + v.z * (self.zmax - self.zmin)

		p = geo.Point.from_arr(v)
		Ns = geo.normalize(self.o2w(geo.Normal(p.x, p.y, p.z)))
		if self.ro:
			Ns *= -1.
		return [self.o2w(p), Ns]

	# """
	# Not account for partial sphere
	# """
	# p = geo.Point.fromVector(radius * uniform_sample_sphere(u1, u2))
	# Ns = normalize(self.o2w(geo.Normal(p.x, p.y, p.z)))
	# if self.ro:
	# 	Ns *= -1.
	# return [self.o2w(p), Ns]

	def refine(self) -> ['Shape']:
		"""
		If `Shape` cannot intersect,
		return a refined subset
		"""
		raise NotImplementedError('Intersecable shapes are not refineable')

	def sample_p(self, pnt: 'geo.Point', u1: FLOAT, u2: FLOAT) -> ['geo.Point', 'geo.Normal']:
		"""
		uniformly sample the sphere
		visible (of certain solid angle)
		to the point
		"""
		# compute coords for sampling
		ctr = self.o2w(geo.Point(0., 0., 0.))
		wc = geo.normalize(ctr - pnt)
		_, wc_x, wc_y = geo.coordinate_system(wc)

		# sample uniformly if p is inside
		if pnt.sq_dist(ctr) - self.radius * self.radius < EPS:
			return self.sample(u1, u2)

		# sample inside subtended cone
		st_max_sq = self.radius * self.radius / pnt.sq_dist(ctr)
		ct_max = np.sqrt(max(0., 1. - st_max_sq))

		from pytracer.montecarlo import uniform_sample_cone
		r = geo.Ray(pnt, uniform_sample_cone(u1, u2, ct_max, wc_x, wc_y, wc), EPS)

		hit, thit, _, _ = self.intersect(r)

		if not hit:
			thit = (ctr - pnt).dot(geo.normalize(r.d))

		ps = r(thit)
		ns = geo.Normal.from_arr(geo.normalize(ps - ctr))
		if self.ro:
			ns *= -1.

		return [ps, ns]

	def pdf_p(self, pnt: 'geo.Point', wi: 'geo.Vector') -> FLOAT:
		ctr = self.o2w(geo.Point(0., 0., 0.))
		# return uniform weight if inside
		if pnt.sq_dist(ctr) - self.radius * self.radius < EPS:
			return super().pdf_p(pnt, wi)

		# general weight
		st_max_sq = self.radius * self.radius / pnt.sq_dist(ctr)
		ct_max = np.sqrt(max(0., 1. - st_max_sq))

		from pytracer.montecarlo import uniform_cone_pdf
		return uniform_cone_pdf(ct_max)

	def intersect(self, r: 'geo.Ray') -> [bool, FLOAT, FLOAT, 'geo.DifferentialGeometry']:
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
		A = ray.d.sq_length()
		B = 2 * ray.d.dot(ray.o)
		C = ray.o.sq_length() - self.radius * self.radius

		D = B * B - 4. * A * C
		if D <= 0. + EPS:
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

		# sphere hit position
		phit = ray(thit)
		if phit.x == 0. and phit.y == 0.:
			phit.x = EPS * self.radius
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0.:
			phi += 2. * np.pi

		# test intersection against clipping params
		if (self.zmin > -self.radius and phit.z < self.zmin) or \
				(self.zmax < self.radius and phit.z > self.zmax) or \
						phi > self.phiMax:
			if thit == t1 or t1 > ray.maxt:
				return [False, None, None, None]

			# try again with t1
			thit = t1
			phit = ray(thit)
			if phit.x == 0. and phit.y == 0.:
				phit.x = EPS * self.radius
			phi = np.arctan2(phit.y, phit.x)
			if phi < 0.:
				phi += 2. * np.pi
			if (self.zmin > -self.radius and phit.z < self.zmin) or \
					(self.zmax < self.radius and phit.z > self.zmax) or \
							phi > self.phiMax:
				return [False, None, None, None]

		# otherwise ray hits the sphere
		# initialize the differential structure
		u = phi / self.phiMax
		theta = np.arccos(np.clip(phit.z / self.radius, -1., 1.))
		delta_theta = self.thetaMax - self.thetaMin
		v = (theta - self.thetaMin) * delta_theta

		# find derivatives
		dpdu = geo.Vector(-self.phiMax * phit.y, self.phiMax * phit.x, 0.)
		zrad = np.sqrt(phit.x * phit.x + phit.y * phit.y)
		inv_zrad = 1. / zrad
		cphi = phit.x * inv_zrad
		sphi = phit.y * inv_zrad
		dpdv = delta_theta \
		       * geo.Vector(phit.z * cphi, phit.z * sphi, -self.radius * np.sin(theta))

		# derivative of Normals
		# given by Weingarten Eqn
		d2pduu = -self.phiMax * self.phiMax * geo.Vector(phit.x, phit.y, 0.)
		d2pduv = delta_theta * phit.z * self.phiMax * geo.Vector(-sphi, cphi, 0.)
		d2pdvv = -delta_theta * delta_theta * geo.Vector(phit.x, phit.y, phit.z)

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
		dg = geo.DifferentialGeometry(o2w(phit), o2w(dpdu), o2w(dpdv), o2w(dndu), o2w(dndv), u, v, self)
		return True, thit, EPS * thit, dg

	def intersect_p(self, r: 'geo.Ray') -> bool:
		# transform ray to object space
		ray = self.w2o(r)

		# solve quad eqn
		A = ray.d.sq_length()
		B = 2 * ray.d.dot(ray.o)
		C = ray.o.sq_length() - self.radius * self.radius

		D = B * B - 4. * A * C
		if D <= 0. + EPS:
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

		phit = ray(thit)
		print("Type phit: {}\nphit.x: {}\nphit.y: {}\n".format(type(phit), type(phit.x), type(phit.y)))
		if phit.x == 0. and phit.y == 0.:
			phit.x = EPS * self.radius
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0.:
			phi += 2. * np.pi

		# test intersection against clipping params
		if (self.zmin > -self.radius and phit.z < self.zmin) or \
				(self.zmax < self.radius and phit.z > self.zmax) or \
						phi > self.phiMax:
			if thit == t1 or t1 > ray.maxt:
				return False

			# try again with t1
			thit = t1
			phit = ray(thit)
			if phit.x == 0. and phit.y == 0.:
				phit.x = EPS * self.radius
			phi = np.arctan2(phit.y, phit.x)
			if phi < 0.:
				phi += 2. * np.pi
			if (self.zmin > -self.radius and phit.z < self.zmin) or \
					(self.zmax < self.radius and phit.z > self.zmax) or \
							phi > self.phiMax:
				return False

		return True

	def area(self) -> FLOAT:
		return self.phiMax * self.radius * (self.zmax - self.zmin)
