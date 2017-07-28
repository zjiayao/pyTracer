'''
shape.py

The base class of shapes.
All `Shape`s is desgined in the
object coordinate system.

Created by Jiayao on July 27, 2017
'''
import numpy as np
from abc import ABCMeta, abstractmethod   
from src.core.pytracer import *
from src.core.geometry import *
from src.core.transform import *
from src.core.diffgeom import *
'''
imp.reload(src.core.pytracer)
imp.reload(src.core.geometry)
imp.reload(src.core.transform)
imp.reload(src.core.shape)
from src.core.pytracer import *
from src.core.geometry import *
from src.core.transform import *
from src.core.shape import *
'''

class Shape(object):
	"""
	Shape Class

	Base class of shapes.
	"""
	__metaclass__ = ABCMeta
	next_shapeId = 1

	def __init__(self, o2w: 'Transform', w2o: 'Transform',
				 ro: bool):
		self.o2w = o2w
		self.w2o = w2o
		self.ro = ro
		self.transform_swaps_handedness = o2w.swaps_handedness()

		self.shapeId = Shape.next_shapeId
		Shape.next_shapeId += 1

	def __repr__(self):
		return "{}\nInstance Count: {}\nShape Id: {}" \
			.format(self.__class__, Shape.next_shapeId, self.shapeId)

	@abstractmethod
	def object_bound(self) -> 'BBox':
		raise NotImplementedError('unimplemented Shape.object_bound() method called') 

	def world_bound(self) -> 'BBox':
		return self.o2w(self.object_bound())

	def can_intersect(self) -> bool:
		return True

	@abstractmethod
	def refine(self) -> 'Shape':
		'''
		If `Shape` cannot intersect,
		return a refined subset
		'''
		raise NotImplementedError('Intersecable shapes are not refineable')

	@abstractmethod
	def intersect(self, r: 'Ray') -> (bool, FLOAT, FLOAT, 'DifferentialGeometry'):
		raise NotImplementedError('unimplemented Shape.intersect() method called') 
		
	@abstractmethod
	def intersectP(self, r: 'Ray') -> bool:
		raise NotImplementedError('unimplemented Shape.intersectP() method called') 

	def get_shading_geometry(self, o2w: 'Transform',
			dg: 'DifferentialGeometry'):
		return dg.copy()

	@abstractmethod
	def area(self) -> FLOAT:
		raise NotImplementedError('unimplemented Shape.area() method called') 


class TriangleMesh(Shape):
	'''
	TriangleMesh Class

	Subclasses `Shape` and is used
	to model trianglular meshes.
	'''	
	def __init__(self, o2w: 'Transform', w2o :'Transform',
			ro: bool, ht: FLOAT, r: FLOAT, ri: FLOAT, pm: FLOAT):
		super().__init__(o2w, w2o, ro)
		self.height = ht
		self.radius = r
		self.inner_radius = ri
		self.phiMax = np.deg2rad(np.clip(pm, 0., 360.))

	def __repr__(self):
		return "{}\nRadius: {}\nInner Radius: {}" \
			.format(self.__class__, self.radius, self.inner_radius)

	def object_bound(self) -> 'BBox':
		return BBox(Point(-self.radius, -self.radius, self.height),
					Point(self.radius, self.radius, self.height))

	def intersect(self, r: 'Ray') -> (bool, FLOAT, FLOAT, 'DifferentialGeometry'):
		'''
		Returns:
		 - bool: specify whether intersects
		 - tHit: hit point param
		 - rEps: error tolerance
		 - dg: DifferentialGeometry object
		'''
		# transform ray to object space
		ray = self.w2o(r)

		# parallel ray has not intersection
		if np.fabs(ray.d.z) < EPS:
			return (False, None, None, None)

		thit = (self.height - ray.o.z) / ray.d.z
		if thit < ray.mint or thit > ray.maxt:
			return (False, None, None, None)

		phit = ray(thit)

		# check radial distance
		dt2 = phit.x * phit.x + phit.y * phit.y
		if dt2 > self.radius * self.radius or \
				dt2 < self.inner_radius * self.inner_radius:
			return (False, None, None, None)

		# check angle
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0:
			phi += 2. * np.pi

		if phi > self.phiMax:
			return (False, None, None, None)

		# otherwise ray hits the disk
		# initialize the differential structure
		u = phi / self.phiMax
		v = 1. - ((np.sqrt(dt2 - self.inner_radius)) /
				  (self.radius - self.inner_radius))

		# find derivatives
		dpdu = Vector(-self.phiMax * phit.y, self.phiMax * phit.x, 0.)
		dpdv = Vector(-phit.x / (1. - v), -phit.y / (1. - v), 0.)
		dpdu *= self.phiMax * INV_2PI
		dpdv *= (self.radius - self.inner_radius) / self.radius
		
		# derivative of Normals
		dndu = Normal(0., 0., 0.,)
		dndv = Normal(0., 0., 0.,)

		o2w = self.o2w
		dg = DifferentialGeometry(o2w(phit), o2w(dpdu), o2w(dpdv),
								  o2w(dndu), o2w(dndv), u, v, self)
		return True, thit, EPS * thit, dg


	def intersectP(self, r: 'Ray') -> bool:	
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






	
class Sphere(Shape):
	'''
	Sphere Class

	Subclasses `Shape` and is used
	to model possibly partial Sphere.
	'''
	def __init__(self, o2w: 'Transform', w2o :'Transform',
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

	def object_bound(self) -> 'BBox':
		return BBox(Point(-self.radius, -self.radius, self.zmin),
					Point(self.radius, self.radius, self.zmax))

	def intersect(self, r: 'Ray') -> (bool, FLOAT, FLOAT, 'DifferentialGeometry'):
		'''
		Returns:
		 - bool: specify whether intersects
		 - tHit: hit point param
		 - rEps: error tolerance
		 - dg: DifferentialGeometry object
		'''
		# transform ray to object space
		ray = self.w2o(r)

		# solve quad eqn
		A = ray.d.sq_length()
		B = 2 * ray.d.dot(ray.o)
		C = ray.o.sq_length() - self.radius * self.radius

		D = B * B - 4. * A * C
		if D <= 0.:
			return (False, None, None, None)

		# validate solutions
		[t0, t1] = np.roots([A,B,C])
		if t0 > t1:
			t0, t1 = t1, t0
		if t0 > ray.maxt or t1 < ray.mint:
			return (False, None, None, None)
		thit = t0
		if t0 < ray.mint:
			thit = t1
			if thit > ray.maxt:
				return (False, None, None, None)

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
				return (False, None, None, None)

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
				return (False, None, None, None)

		# otherwise ray hits the sphere
		# initialize the differential structure
		u = phi / self.phiMax
		theta = np.arccos(np.clip(phit.z / self.radius, -1., 1.))
		delta_theta = self.thetaMax - self.thetaMin
		v = (theta - self.thetaMin) * delta_theta

		# find derivatives
		dpdu = Vector(-self.phiMax * phit.y, self.phiMax * phit.x, 0.)
		zrad = np.sqrt(phit.x * phit.x + phit.y * phit.y)
		inv_zrad = 1. / zrad
		cphi = phit.x * inv_zrad
		sphi = phit.y * inv_zrad
		dpdv = delta_theta \
					* Vector(phit.z * cphi, phit.z * sphi, -self.radius * np.sin(theta))

		# derivative of Normals
		# given by Weingarten Eqn
		d2pduu = -self.phiMax * self.phiMax * Vector(phit.x, phit.y, 0.)
		d2pduv = delta_theta * phit.z * self.phiMax * Vector(-sphi, cphi, 0.)
		d2pdvv = -delta_theta * delta_theta * Vector(phit.x, phit.y, phit.z)

		# fundamental forms
		E = dpdu.dot(dpdu)
		F = dpdu.dot(dpdv)
		G = dpdv.dot(dpdv)
		N = normalize(dpdu.cross(dpdv))
		e = N.dot(d2pduu)
		f = N.dot(d2pduv)
		g = N.dot(d2pdvv)

		invEGFF = 1. / (E * G - F * F)
		dndu = Normal.fromVector((f * F - e * G) * invEGFF * dpdu +
								 (e * F - f * E) * invEGFF * dpdv)
		dndv = Normal.fromVector((g * F - f * G) * invEGFF * dpdu +
								 (f * F - g * E) * invEGFF * dpdv)

		o2w = self.o2w
		dg = DifferentialGeometry(o2w(phit), o2w(dpdu), o2w(dpdv),
								  o2w(dndu), o2w(dndv), u, v, self)
		return True, thit, EPS * thit, dg


	def intersectP(self, r: 'Ray') -> bool:	
		# transform ray to object space
		ray = self.w2o(r)

		# solve quad eqn
		A = ray.d.sq_length()
		B = 2 * ray.d.dot(ray.o)
		C = ray.o.sq_length() - self.radius * self.radius

		D = B * B - 4. * A * C
		if D <= 0.:
			return False

		# validate solutions
		[t0, t1] = np.roots([A,B,C])
		if t0 > t1:
			t0, t1 = t1, t0
		if t0 > ray.maxt or t1 < ray.mint:
			return False
		thit = t0
		if t0 < ray.mint:
			thit = t1
			if thit > ray.maxt:
				return False

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

class Cylinder(Shape):
	'''
	Cylinder Class

	Subclasses `Shape` and is used
	to model possibly partial Cylinder.
	'''	
	def __init__(self, o2w: 'Transform', w2o :'Transform',
			ro: bool, rad: FLOAT, z0: FLOAT, z1: FLOAT, pm: FLOAT):
		super().__init__(o2w, w2o, ro)
		self.radius = rad
		self.zmin = min(z0, z1)
		self.zmax = max(z0, z1)
		self.phiMax = np.deg2rad(np.clip(pm, 0., 360.))

	def __repr__(self):
		return "{}\nRadius: {}".format(self.__class__, self.radius)

	def object_bound(self) -> 'BBox':
		return BBox(Point(-self.radius, -self.radius, self.zmin),
					Point(self.radius, self.radius, self.zmax))

	def intersect(self, r: 'Ray') -> (bool, FLOAT, FLOAT, 'DifferentialGeometry'):
		'''
		Returns:
		 - bool: specify whether intersects
		 - tHit: hit point param
		 - rEps: error tolerance
		 - dg: DifferentialGeometry object
		'''
		# transform ray to object space
		ray = self.w2o(r)

		# solve quad eqn
		A = ray.d.x * ray.d.x + ray.d.y * ray.d.y
		B = 2 * (ray.d.x * ray.o.x + ray.d.y * ray.o.y)
		C = ray.o.x * ray.o.x + ray.o.y * ray.o.y - self.radius * self.radius

		D = B * B - 4. * A * C
		if D <= 0.:
			return (False, None, None, None)

		# validate solutions
		[t0, t1] = np.roots([A,B,C])
		if t0 > t1:
			t0, t1 = t1, t0
		if t0 > ray.maxt or t1 < ray.mint:
			return (False, None, None, None)
		thit = t0
		if t0 < ray.mint:
			thit = t1
			if thit > ray.maxt:
				return (False, None, None, None)

		# cylinder hit position
		phit = ray(thit)
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0.:
			phi += 2. * np.pi

		# test intersection against clipping params
		if phit.z < self.zmin or phit.z > self.zmax or phi > self.phiMax:
			if thit == t1 or t1 > ray.maxt:
				return (False, None, None, None)

			# try again with t1
			thit = t1
			phit = ray(thit)
			phi = np.arctan2(phit.y, phit.x)
			if phi < 0.:
				phi += 2. * np.pi

			if phit.z < self.zmin or phit.z > self.zmax or phi > self.phiMax:
				if thit == t1 or t1 > ray.maxt:
					return (False, None, None, None)

		# otherwise ray hits the cylinder
		# initialize the differential structure
		u = phi / self.phiMax
		v = (phit.z - self.zmin) / (self.thetaMax - self.thetaMin)

		# find derivatives
		dpdu = Vector(-self.phiMax * phit.y, self.phiMax * phit.x, 0.)
		dpdv = Vector(0., 0., self.zmax - self.zmin)
		
		# derivative of Normals
		# given by Weingarten Eqn
		d2pduu = -self.phiMax * self.phiMax * Vector(phit.x, phit.y, 0.)
		d2pduv = Vector(0., 0., 0.)
		d2pdvv = Vector(0., 0., 0.)

		# fundamental forms
		E = dpdu.dot(dpdu)
		F = dpdu.dot(dpdv)
		G = dpdv.dot(dpdv)
		N = normalize(dpdu.cross(dpdv))
		e = N.dot(d2pduu)
		f = N.dot(d2pduv)
		g = N.dot(d2pdvv)

		invEGFF = 1. / (E * G - F * F)
		dndu = Normal.fromVector((f * F - e * G) * invEGFF * dpdu +
								 (e * F - f * E) * invEGFF * dpdv)
		dndv = Normal.fromVector((g * F - f * G) * invEGFF * dpdu +
								 (f * F - g * E) * invEGFF * dpdv)

		o2w = self.o2w
		dg = DifferentialGeometry(o2w(phit), o2w(dpdu), o2w(dpdv),
								  o2w(dndu), o2w(dndv), u, v, self)
		return True, thit, EPS * thit, dg


	def intersectP(self, r: 'Ray') -> bool:	
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
		[t0, t1] = np.roots([A,B,C])
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

class Disk(Shape):
	'''
	Disk Class

	Subclasses `Shape` and is used
	to model possibly partial Disk.
	'''	
	def __init__(self, o2w: 'Transform', w2o :'Transform',
			ro: bool, ht: FLOAT, r: FLOAT, ri: FLOAT, pm: FLOAT):
		super().__init__(o2w, w2o, ro)
		self.height = ht
		self.radius = r
		self.inner_radius = ri
		self.phiMax = np.deg2rad(np.clip(pm, 0., 360.))

	def __repr__(self):
		return "{}\nRadius: {}\nInner Radius: {}" \
			.format(self.__class__, self.radius, self.inner_radius)

	def object_bound(self) -> 'BBox':
		return BBox(Point(-self.radius, -self.radius, self.height),
					Point(self.radius, self.radius, self.height))

	def intersect(self, r: 'Ray') -> (bool, FLOAT, FLOAT, 'DifferentialGeometry'):
		'''
		Returns:
		 - bool: specify whether intersects
		 - tHit: hit point param
		 - rEps: error tolerance
		 - dg: DifferentialGeometry object
		'''
		# transform ray to object space
		ray = self.w2o(r)

		# parallel ray has not intersection
		if np.fabs(ray.d.z) < EPS:
			return (False, None, None, None)

		thit = (self.height - ray.o.z) / ray.d.z
		if thit < ray.mint or thit > ray.maxt:
			return (False, None, None, None)

		phit = ray(thit)

		# check radial distance
		dt2 = phit.x * phit.x + phit.y * phit.y
		if dt2 > self.radius * self.radius or \
				dt2 < self.inner_radius * self.inner_radius:
			return (False, None, None, None)

		# check angle
		phi = np.arctan2(phit.y, phit.x)
		if phi < 0:
			phi += 2. * np.pi

		if phi > self.phiMax:
			return (False, None, None, None)

		# otherwise ray hits the disk
		# initialize the differential structure
		u = phi / self.phiMax
		v = 1. - ((np.sqrt(dt2 - self.inner_radius)) /
				  (self.radius - self.inner_radius))

		# find derivatives
		dpdu = Vector(-self.phiMax * phit.y, self.phiMax * phit.x, 0.)
		dpdv = Vector(-phit.x / (1. - v), -phit.y / (1. - v), 0.)
		dpdu *= self.phiMax * INV_2PI
		dpdv *= (self.radius - self.inner_radius) / self.radius
		
		# derivative of Normals
		dndu = Normal(0., 0., 0.,)
		dndv = Normal(0., 0., 0.,)

		o2w = self.o2w
		dg = DifferentialGeometry(o2w(phit), o2w(dpdu), o2w(dpdv),
								  o2w(dndu), o2w(dndv), u, v, self)
		return True, thit, EPS * thit, dg


	def intersectP(self, r: 'Ray') -> bool:	
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




