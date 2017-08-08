'''
primitive.py

This module is part of the pyTracer, which
defines `Primitive` classes.

v0.0
Created by Jiayao on July 30, 2017
'''
'''
import src.core.pytracer
import src.core.geometry
import src.core.transform
import src.core.shape
import src.core.primitive
imp.reload(src.core.pytracer)
imp.reload(src.core.geometry)
imp.reload(src.core.transform)
imp.reload(src.core.shape)
imp.reload(src.core.primitive)
from src.core.pytracer import *
from src.core.geometry import *
from src.core.transform import *
from src.core.shape import *
from src.core.primitive import *
'''
import numpy as np
from abc import ABCMeta, abstractmethod  
import threading
from src.core.pytracer import *
from src.core.geometry import *
from src.core.transform import *
from src.core.shape import *

class Intersection(object):
	def __init__(self, dg: 'DifferentialGeometry', pr: 'Primitive',
			w2o: 'Transform', o2w: 'Transform', sh_id: INT, pr_id: INT, rEps: FLOAT):
		self.dg = dg
		self.primitive = pr
		self.w2o = w2o
		self.o2w = o2w
		self.shapeId = sh_id
		self.primitiveId = pr_id
		self.rEps = rEps

	def get_BSDF(self, ray: 'RayDifferential') -> 'BSDF':
		self.dg.compute_differential(ray)
		return self.primitive.get_BSDF(dg, o2w)

	def get_BSSRDF(self, ray: 'RayDifferential') -> 'BSSRDF':
		self.dg.compute_differential(ray)
		return self.primitive.get_BSSRDF(dg, o2w)

	def le(self, w: 'Vector') -> 'Spectrum':
		'''
		le()

		Compute the emitted radiance
		at a surface point intersected by
		a ray.
		'''
		area = self.primitive.get_area_light()
		if area is None:
			return Spectrum(0.)
		return area.l(self.dg.p, self.dg.nn, w)


class Primitive(object, metaclass=ABCMeta):

	__primitiveId = 0
	next_primitiveId = 1
	def __init__(self):
		Primitive.__primitiveId = Primitive.next_primitiveId
		GP.next_primitiveId += 1

	def __repr__(self):
		return "{}\nPrimitive ID: {}\nNext Primitive ID: {}" \
			.format(self.__class__, Primitive.__primitiveId, Primitive.next_primitiveId)

	@abstractmethod
	def world_bound(self) -> BBox:
		raise NotImplementedError('{}.world_bound(): Not implemented'.format(self.__class__))

	@abstractmethod
	def can_intersect(self) -> bool:
		raise NotImplementedError('{}.can_intersect(): Not implemented'.format(self.__class__))

	@abstractmethod
	def intersect(self, r: 'Ray') -> [bool, 'Intersection']:
		raise NotImplementedError('{}.intersect(): Not implemented'.format(self.__class__))

	@abstractmethod
	def intersectP(self, r : 'Ray') -> bool:
		raise NotImplementedError('{}.intersectP(): Not implemented'.format(self.__class__))

	@abstractmethod
	def refine(self, refined: ['Primitive']):
		raise NotImplementedError('{}.refine(): Not implemented'.format(self.__class__))

	@abstractmethod
	def get_area_light(self):
		raise NotImplementedError('{}.get_area_light(): Not implemented'.format(self.__class__))

	@abstractmethod
	def get_BSDF(self, dg: 'DifferentialGeometry', o2w: 'Transform') -> 'BSDF':
		raise NotImplementedError('{}.get_BSDF(): Not implemented'.format(self.__class__))

	@abstractmethod
	def get_BSSRDF(self, dg: 'DifferentialGeometry', o2w: 'Transform') -> 'BSSRDF':
		raise NotImplementedError('{}.get_BSSRDF(): Not implemented'.format(self.__class__))

	def full_refine(self, refined: ['Primitive']):
		todo = [self]
		while len(todo) > 0:
			prim = todo[-1]
			del todo[-1]
			if prim.can_intersect():
				refined.append(prim)
			else:
				prim.refine(todo)
	
	
# Shapes to be rendered directly
class GeometricPrimitive(Primitive):
	def __init__(self, s: 'Shape', m: 'Material', a: 'AreaLight' = None):
		super().__init__()
		self.shape = s
		self.material = m
		self.areaLight = a

	def __repr__(self):
		return super().__repr__(self) + '\n{}'.format(self.shape)

	def intersect(self, r: 'Ray') -> [bool, 'Intersection']:
		is_intersect, thit, rEps, dg = self.shape.intersect(r)
		if not is_intersect:
			return [False, None]

		isect = Intersection(dg, self, self.shape.w2o, self.shape.o2w,
			self.shape.shapeId, self.primitiveId, rEps)
		r.maxt = thit
		return [True, isect]

	def intersectP(self, r: 'Ray') -> bool:
		return self.shape.intersectP(r)		

	def world_bound(self) -> 'BBox':
		return self.shape.world_bound()

	def can_intersect(self) -> bool:
		return self.shape.can_intersect()

	def refine(self, refined: ['Primitive']):
		r = self.shape.refine()
		for sh in r:
			refined.append(GeometricPrimitive(sh, self.material, self.areaLight))

	def get_area_light(self):
		return self.areaLight

	def get_BSDF(self, dg: 'DifferentialGeometry', o2w: 'Transform'):
		dgs = self.shape.get_shading_geometry(o2w, dg)
		return self.material.get_BSDF(dg, dgs)

	def get_BSSRDF(self, dg: 'DifferentialGeometry', o2w: 'Transform'):
		dgs = self.shape.get_shading_geometry(o2w, dg)
		return self.material.get_BSSRDF(dg, dgs)


# Shapes with animated transfomration and object instancing
class TransformedPrimitive(Primitive):
	def __init__(self, prim: 'Primitive', w2p: 'AnimatedTransform'):
		super().__init__()
		self.primitive = prim
		self.w2p = w2p

	def __repr__(self):
		return super().__repr__(self) + '\n{}'.format(self.prim)

	def intersect(self, r: 'Ray') -> [bool, 'Intersection']:
		w2p = self.w2p.interpolate(r.time)
		ray = w2p(r)
		is_intersect, isect = self.primitive.intersect(ray)
		if not is_intersect:
			return [False, None]
		r.maxt = ray.maxt
		isect.primitiveId = self.primitiveId

		if not w2p.is_identity():
			isect.w2o = isect.w2o * w2p
			isect.o2w = inverse(isec.w2o)

			p2w = inverse(w2p)
			dg = isect.dg
			dg.p = p2w(dg.p)
			dg.nn = normalize(p2w(dg.nn))
			dg.dpdu = p2w(dg.dpdu)
			dg.dpdv = p2w(dg.dpdv)
			dg.dndu = p2w(dg.dndu)
			dg.dndv = p2w(dg.dndv)

		return True, isect

	def intersectP(self, r: 'Ray') -> bool:
		return self.primitive.intersectP(self.w2p(r))		

	def world_bound(self) -> 'BBox':
		return self.w2p.motion_bounds(self.primitive.world_bound(), True)

	def can_intersect(self) -> bool:
		return self.shape.can_intersect()

	def refine(self, refined: ['Primitive']):
		r = self.shape.refine()
		for sh in r:
			refined.append(GeometricPrimitive(sh, self.material, self.areaLight))


# Aggregates
class Aggregate(Primitive):
	def __init__(self):
		super().__init__()

	def get_area_light(self):
		raise RuntimeError('{}.get_area_light(): Should not be called'.format(self.__class__))

	def get_BSDF(self, dg: 'DifferentialGeometry', o2w: 'Transform'):
		raise RuntimeError('{}.get_BSDF(): Should not be called'.format(self.__class__))

	def get_BSSRDF(self, dg: 'DifferentialGeometry', o2w: 'Transform'):
		raise RuntimeError('{}.get_BSSRDF(): Should not be called'.format(self.__class__))

class Voxel():
	def __init__(self, op: ['Primitive']):
		self.all_can_intersect = False
		self.primitives.extend(op)

	def __repr__(self):
		return "{}\nPrimitives: {}".format(self.__class__, len(self.primitives))

	def add_primitive(prim: 'Primitive'):
		self.primitives.append(prim)

	def intersect(self, ray: 'Ray', lock) -> [bool, 'Intersection']:
		# refine primitives if needed
		if not self.all_can_intersect:
			lock.acquire()
			for i, prim in enumerate(self.primitives):
				# refine if necessary
				if not prim.can_intersect():
					p = []
					prim.full_refine(p)
					if len(p) == 1:
						self.primitives[i] = p[0]
					else:
						self.primitives[i] = GridAccel(p, False)

			self.all_can_intersect = True
			lock.release()

		# loop over
		# no data corrpution?
		anyHit = False
		for prim in self.primitives:
			hit, isect = prim.intersect(ray)
			if hit:
				anyHit = True
		return [anyHit, isect] # weird of returning isect

	def intersectP(self, ray: 'Ray', lock) -> bool:
		# refine primitives if needed
		if not self.all_can_intersect:
			lock.acquire()
			for i, prim in enumerate(self.primitives):
				# refine if necessary
				if not prim.can_intersect():
					p = []
					prim.full_refine(p)
					if len(p) == 1:
						self.primitives[i] = p[0]
					else:
						self.primitives[i] = GridAccel(p, False)

			self.all_can_intersect = True
			lock.release()

		# loop over
		# no data corrpution?
		for prim in self.primitives:
			if prim.intersectP(ray):
				return True
		return False


# Grid Accelerator
class GridAccel(Aggregate):
	def __init__(self, p: 'np.ndarray', refineImm: bool):

		if refineImm:
			self.primitives = []
			for prim in p:
				prim.full_refine(self.primitives)
		else:
			self.primitives = p

		# compute bounds and choose grid resolution
		self.bounds = BBox()
		self.nVoxels = [0, 0, 0]
		for prim in self.primitives:
			self.bounds.union(prim.world_bound())
		
		delta = bounds.pMax - bounds.pMin
		maxAxis = bounds.maximum_extent()
		invMaxWidth = 1. / delta[maxAxis]
		cubeRoot = 3. * np.pow(len(self.primitive), 1./3.)
		voxels_per_unit_dist = cubeRoot * invMaxWidth
		
		self.width = Vector()
		self.invWidth = Vector()

		for axis in range(3):
			nVoxels[axis] = np.int(delta[axis] * voxels_per_unit_dist)
			nVoxels[axis] = np.clip(nVoxels[axis], 1., 64.)

			self.width[axis] = delta[axis] / nVoxels[axis]
			self.invWidth[axis] = 0. if self.width[axis] == 0. else 1. / self.width[axis]

		nv = np.prod(nVoxels)
		self.voxels = np.full(nv, None)

		# add primitives to voxels
		for prim in self.primitives:
			pb = prim.world_bound()
			vmin = [pos2voxel(pb.pMin, axis) for axis in range(3)]
			vmax = [pos2voxel(pb.pMax, axis) for axis in range(3)]

			for z in range(vmin[2], vmax[2] + 1):
				for y in range(vmin[1], vmax[1] + 1):
					for x in range(vimn[0], vmax[0] + 1):
						o = self.offset(x, y, z)

						if self.voxels[o] is None:
							# new voxel
							voxels[o] = Voxel(prim)

						else:
							# add primitive
							voxels[o].add_primitive(prim)

		# create mutex for grid
		self.lock = threading.Lock()

		def pos2voxel(self, P: 'Point', axis: INT) -> INT:
			v = np.int((P[axis] - self.bounds.pMin[axis]) * self.invWidth[axis])
			return np.clip(v, 0, self.nVoxels[axis] - 1)

		def voxel2pos(self, p: INT, axis: INT) -> FLOAT:
			return self.bounds.pMin[axis] + p * self.width[axis]

		def offset(self, x: INT, y: INT, z: INT) -> INT:
			return z * self.nVoxels[0] * self.nVoxels[1] + y * self.nVoxels[0] + x

		def world_bound(self) -> 'BBox':
			return self.bounds

		def can_intersect(self) -> bool:
			return True

		def intersect(self, ray: 'Ray') -> [bool, 'Intersection']:
			# Check ray aginst overall bounds
			if bounds.inside(ray(ray.mint)):
				rayT = ray.mint
			elif not bounds.intersectP(ray):
				return [False, None]
			grid_intersect = ray(rayT)

			# Difference between Bresenham's Line Drawing:
			# find all voxels that ray passes through
			# digital differential analyzer
			# Set up 3D DDA for Ray

			pos = [pos2voxel(axis) for axis in range(3)]
			next_crossing = [rayT + voxel2pos(pos[axis]+1, axis) - grid_intersect[axis] / ray.d[axis]\
									for axis in range(3)]
			delta_t = self.width / ray.d
			step = [1,1,1]
			out = self.nVoxels.copy()
			for axis in range(3):
				# compute current voxel
				if ray.d[axis] < 0:
					# ray with neg. direction for stepping
					delta_t[axis] = -delta_t[axis]
					step[axis] = out[axis] = -1

			# walk through grid
			anyHit = False
			while True:
				voxel = self.voxels[self.offset(pos[0], pos[1], pos[2])]
				if voxel is not None:
					hit, isect = voxel.intersect(ray, self.lock)
					anyHit |= hit
				# next voxel
				step_axis = np.argmin(next_crossing)

				if ray.maxt < next_crossing[step_axis]:
					break
				pos[step_axis] += step[step_axis]
				if pos[step_axis] == out[step_axis]:
					break
				next_crossing[step_axis] += delta_t[step_axis]

			return [hit, isect]

		def intersectP(self, ray: 'Ray') -> bool:
			# Check ray aginst overall bounds
			if bounds.inside(ray(ray.mint)):
				rayT = ray.mint
			elif not bounds.intersectP(ray):
				return False
			grid_intersect = ray(rayT)

			# Difference between Bresenham's Line Drawing:
			# find all voxels that ray passes through
			# digital differential analyzer
			# Set up 3D DDA for Ray

			pos = [pos2voxel(axis) for axis in range(3)]
			next_crossing = [rayT + voxel2pos(pos[axis]+1, axis) - grid_intersect[axis] / ray.d[axis]\
									for axis in range(3)]
			delta_t = self.width / ray.d
			step = [1,1,1]
			out = self.nVoxels.copy()
			for axis in range(3):
				# compute current voxel
				if ray.d[axis] < 0:
					# ray with neg. direction for stepping
					delta_t[axis] = -delta_t[axis]
					step[axis] = out[axis] = -1

			# walk through grid
			anyHit = False
			while True:
				voxel = self.voxels[self.offset(pos[0], pos[1], pos[2])]
				if voxel is not None:
					hit, isect = voxel.intersectP(ray, self.lock)
					anyHit |= hit
				# next voxel
				step_axis = np.argmin(next_crossing)

				if ray.maxt < next_crossing[step_axis]:
					break
				pos[step_axis] += step[step_axis]
				if pos[step_axis] == out[step_axis]:
					break
				next_crossing[step_axis] += delta_t[step_axis]

			return anyHit

# BVH Accelerator
class BVHAccel(Aggregate):
	class BVHPrimInfo():
		def __init__(self, pn: INT, b: BBox):
			self.primitiveId = pn
			self.bunds = b
			self.centroid = .5 * b.pMin + .5 * b.pMax

		def __repr__(self):
			return "{}\nCentroid: {}".format(self.__class__, self.centroid)

	def __init__(self, p: ['Primitive'], mp: INT, algo: str):
		self.max_prims_in_node = min(mp, 255)
		self.primitives = []
		for i, prim in enumerate(p):
			p[i].full_refine(primitives)

		if algo == "sah" or algo == "surface area heuristic":
			self.splitMethod = SPLIT_SAH
		else:
			print("[Warning] src.core.primitive.{}: unknown BVH split method, using sah." \
				.format(self.__class__))
			self.splitMethod = SPLIT_SAH

		if len(self.primitives) == 0:
			self.nodes = None
			return
		# construct BVH
		## init build_data
		build_data = np.empty(len(self.primitives), dtype=object)
		for i, prim in enumerate(self.primitives):
			bbox = prim.world_bound()
			build_data[i] = BVHPrimInfo(i, bbox)
			root = 


		## build BVH tree recursively


		## representation for DFS





