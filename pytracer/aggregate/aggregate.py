"""
aggregate.py

Implementation of aggregates.

v0.0
Created by Jiayao on July 30, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
import threading
from pytracer import *
import pytracer.geometry as geo
import pytracer.transform as trans
from pytracer.aggregate.primitive import Primitive
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from pytracer.aggregate import Intersection


__all__ = ['Aggregate', 'Voxel', 'GridAccel']


# Aggregates
class Aggregate(Primitive):
	def __init__(self):
		super().__init__()

	def get_area_light(self):
		raise RuntimeError('{}.get_area_light(): Should not be called'.format(self.__class__))

	def get_bsdf(self, dg: 'geo.DifferentialGeometry', o2w: 'trans.Transform'):
		raise RuntimeError('{}.get_bsdf(): Should not be called'.format(self.__class__))

	def get_bssrdf(self, dg: 'geo.DifferentialGeometry', o2w: 'trans.Transform'):
		raise RuntimeError('{}.get_bssrdf(): Should not be called'.format(self.__class__))


class Voxel(object):
	"""Voxel Class"""
	def __init__(self, op: ['Primitive']):
		self.all_can_intersect = False
		self.primitives = op.copy()

	def __repr__(self):
		return "{}\nPrimitives: {}".format(self.__class__, len(self.primitives))

	def add_primitive(self, prim: 'Primitive'):
		self.primitives.append(prim)

	def intersect(self, ray: 'geo.Ray', lock) -> [bool, 'Intersection']:
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
		any_hit = False
		isect = None
		for prim in self.primitives:
			hit, ist = prim.intersect(ray)
			if hit:
				isect = ist
			if hit:
				any_hit = True
		return [any_hit, isect]  # weird of returning isect

	def intersect_p(self, ray: 'geo.Ray', lock) -> bool:
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
			if prim.intersect_p(ray):
				return True
		return False


# Grid Accelerator
class GridAccel(Aggregate):
	def __init__(self, p: 'np.ndarray', refine_imm: bool):
		super().__init__()
		if refine_imm:
			self.primitives = []
			for prim in p:
				prim.full_refine(self.primitives)
		else:
			self.primitives = p

		# compute bounds and choose grid resolution
		self.bounds = geo.BBox()
		self.n_voxels = [0, 0, 0]

		for prim in self.primitives:
			self.bounds.union(prim.world_bound())

		delta = self.bounds.pMax - self.bounds.pMin
		max_axis = self.bounds.maximum_extent()
		inv_max_width = 1. / delta[max_axis]
		cube_root = 3. * np.power(len(self.primitives), 1. / 3.)
		voxels_per_unit_dist = cube_root * inv_max_width

		self.width = geo.Vector()
		self.invWidth = geo.Vector()

		nVoxels = [0., 0., 0.]

		for axis in range(3):
			nVoxels[axis] = np.int(delta[axis] * voxels_per_unit_dist)
			nVoxels[axis] = np.clip(nVoxels[axis], 1., 64.)

			self.width[axis] = delta[axis] / nVoxels[axis]
			self.invWidth[axis] = 0. if self.width[axis] == 0. else 1. / self.width[axis]

		nv = np.prod(nVoxels).astype(INT)
		self.voxels = np.full(nv, None)

		# add primitives to voxels
		for prim in self.primitives:
			pb = prim.world_bound()
			vmin = [self.pos2voxel(pb.pMin, axis) for axis in range(3)]
			vmax = [self.pos2voxel(pb.pMax, axis) for axis in range(3)]

			for z in range(vmin[2], vmax[2] + 1):
				for y in range(vmin[1], vmax[1] + 1):
					for x in range(vmin[0], vmax[0] + 1):
						o = self.offset(x, y, z)

						if self.voxels[o] is None:
							# new voxel
							self.voxels[o] = Voxel([prim])

						else:
							# add primitive
							self.voxels[o].add_primitive(prim)

		# create mutex for grid
		self.lock = threading.Lock()

	def refine(self, refined: ['Primitive']):
		raise NotImplementedError('{}.refine(): Not implemented'.format(self.__class__))

	def pos2voxel(self, p: 'geo.Point', axis: INT) -> INT:
		v = np.int((p[axis] - self.bounds.pMin[axis]) * self.invWidth[axis])
		return np.clip(v, 0, self.n_voxels[axis] - 1).astype(INT)

	def voxel2pos(self, p: INT, axis: INT) -> FLOAT:
		return self.bounds.pMin[axis] + p * self.width[axis]

	def offset(self, x: INT, y: INT, z: INT) -> INT:
		return z * self.n_voxels[0] * self.n_voxels[1] + y * self.n_voxels[0] + x

	def world_bound(self) -> 'geo.BBox':
		return self.bounds

	def can_intersect(self) -> bool:
		return True

	def intersect(self, ray: 'geo.Ray') -> [bool, 'Intersection']:
		# Check ray aginst overall bounds
		ray_t = 0.
		if self.bounds.inside(ray(ray.mint)):
			ray_t = ray.mint
		elif not self.bounds.intersect_p(ray):
			return [False, None]

		grid_intersect = ray(ray_t)

		# Difference between Bresenham's Line Drawing:
		# find all voxels that ray passes through
		# digital differential analyzer
		# Set up 3D DDA for geo.Ray

		pos = [self.pos2voxel(grid_intersect, axis) for axis in range(3)]
		next_crossing = [ray_t + self.voxel2pos(pos[axis] + 1, axis) - grid_intersect[axis] / ray.d[axis] \
		                 for axis in range(3)]
		delta_t = self.width / ray.d
		step = [1, 1, 1]
		out = self.n_voxels.copy()
		for axis in range(3):
			# compute current voxel
			if ray.d[axis] < 0:
				# ray with neg. direction for stepping
				delta_t[axis] = -delta_t[axis]
				step[axis] = out[axis] = -1

		# walk through grid
		any_hit = False
		isect = None
		while True:
			voxel = self.voxels[self.offset(pos[0], pos[1], pos[2])]
			if voxel is not None:
				hit, ist = voxel.intersect(ray, self.lock)
				if hit:
					isect = ist
				any_hit |= hit
			# next voxel
			step_axis = np.argmin(next_crossing)

			if ray.maxt < next_crossing[step_axis]:
				break
			pos[step_axis] += step[step_axis]
			if pos[step_axis] == out[step_axis]:
				break
			next_crossing[step_axis] += delta_t[step_axis]

		return [any_hit, isect]

	def intersect_p(self, ray: 'geo.Ray') -> bool:
		# Check ray aginst overall bounds
		ray_t = 0.
		if self.bounds.inside(ray(ray.mint)):
			ray_t = ray.mint
		elif not self.bounds.intersect_p(ray):
			return False

		grid_intersect = ray(ray_t)

		# Difference between Bresenham's Line Drawing:
		# find all voxels that ray passes through
		# digital differential analyzer
		# Set up 3D DDA for geo.Ray

		pos = [self.pos2voxel(grid_intersect, axis) for axis in range(3)]
		next_crossing = [ray_t + self.voxel2pos(pos[axis] + 1, axis) - grid_intersect[axis] / ray.d[axis] \
		                 for axis in range(3)]
		delta_t = self.width / ray.d
		step = [1, 1, 1]
		out = self.n_voxels.copy()
		for axis in range(3):
			# compute current voxel
			if ray.d[axis] < 0:
				# ray with neg. direction for stepping
				delta_t[axis] = -delta_t[axis]
				step[axis] = out[axis] = -1

		# walk through grid
		any_hit = False
		while True:
			voxel = self.voxels[self.offset(pos[0], pos[1], pos[2])]
			if voxel is not None:
				hit = voxel.intersect_p(ray, self.lock)
				any_hit |= hit
			# next voxel
			step_axis = np.argmin(next_crossing)

			if ray.maxt < next_crossing[step_axis]:
				break
			pos[step_axis] += step[step_axis]
			if pos[step_axis] == out[step_axis]:
				break
			next_crossing[step_axis] += delta_t[step_axis]

		return any_hit


# TODO
"""
# BVH Accelerator
class BVHAccel(Aggregate):
	class BVHPrimInfo():
		def __init__(self, pn: INT, b: geo.BBox):
			self.primitiveId = pn
			self.bunds = b
			self.centroid = .5 * b.pMin + .5 * b.pMax

		def __repr__(self):
			return "{}\nCentroid: {}".format(self.__class__, self.centroid)

	def __init__(self, p: ['Primitive'], mp: INT, algo: str):
		self.max_prims_in_node = min(mp, 255)
		self.primitives = []
		for i, prim in enumerate(p):
			p[i].full_refine(self.primitives)

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


		## build BVH tree recursively


		## representation for DFS
"""



