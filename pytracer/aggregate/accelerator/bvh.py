"""
bvh.py

pytracer.aggregate.accelerator package

Bounding volume hierarchies

v0.0
Created by Jiayao on August 24, 2017
"""
from __future__ import (division, absolute_import)
from enum import Enum
import numpy as np
from pytracer import (INT, UINT, FLOAT, util)
import pytracer.geometry as geo
from pytracer.aggregate.aggregate import Aggregate
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from pytracer.aggregate import (Primitive, Intersection)

__all__ = ['BVH']


class BVH(Aggregate):
	"""BVH Class"""
	class SplitMethod(Enum):
		MIDDLE = 0
		# EQUAL_COUNTS = 1  # coming soon
		SAH = 2

	class _BVHPrimitive(object):
		def __init__(self, idx: INT, bbox: 'geo.BBox'):
			self.prim_idx = idx
			self.bounds = bbox
			self.centroid = .5 * (bbox.pMin + bbox.pMax)

		def __repr__(self):
			return "{}\nIndex: {}\nCentroid: {}\n".format(self.__class__, self.prim_idx, self.bounds)

	class _BVHNode(object):
		def __init__(self):
			self.children = [None, None]
			self.bounds = None
			self.split_axis = 0
			self.first_offset = 0
			self.n_prim = 0

		def init_leaf(self, first: UINT, n: UINT, b: geo.BBox):
			self.first_offset = first
			self.n_prim = n
			self.bounds = b

		def init_inter(self, axis: UINT, c0: 'BVH._BVHNode', c1: 'BVH._BVHNode'):
			self.children[0] = c0
			self.children[1] = c1
			self.bounds = geo.BBox.Union(c0.bounds, c1.bounds)
			self.split_axis = axis
			self.n_prim = 0
	
	class _LinearNode(object):
		"""Flattened nodes for storage and quick look-up."""
		def __init__(self, bounds: geo.BBox=None, offset: UINT=0, n_prim: UINT=0, axis: UINT=0):
			self.bounds = bounds
			self.offset = offset
			self.n_prim = n_prim
			self.axis = axis

	def __init__(self, p: ['Primitive'], max_prim_per_node: UINT=4, method: str='sah'):
		super().__init__()
		self.primitives = []
		self.max_prim_per_node = max_prim_per_node
		for prim in p:
			prim.full_refine(self.primitives)

		if method.lower() == 'sah':
			self.split_method = BVH.SplitMethod.SAH
		elif method.lower() == 'middle':
			self.split_method = BVH.SplitMethod.MIDDLE
		elif method.lower() == 'equal':
			self.split_method = BVH.SplitMethod.EQUAL_COUNTS
		else:
			util.logging('Error', 'BVH split method unknown, using SAH.')
			self.split_method = BVH.SplitMethod.SAH

		if len(self.primitives) == 0:
			self.nodes = None
			return

		# building BVH
		# init build_data
		build_data = [BVH._BVHPrimitive(i, prim.world_bound()) for i, prim in enumerate(self.primitives)]

		# recursively build BVH
		segment = [0, len(self.primitives), 0]
		ordered_prims = [None] * len(self.primitives)
		root = self._recursive_build(build_data, segment, ordered_prims)

		self.primitives = ordered_prims

		# DFS of BVH
		self.nodes = [BVH._LinearNode() for _ in range(segment[2])]
		self._flatten_tree(root, 0)

	@staticmethod
	def _partition(data: ['BVH._BVHPrimitive'], start: INT, end: INT, mid: FLOAT, dim: INT):
		i = start
		k = end - 1
		while i < k:
			if data[i].centroid[dim] < mid:
				i += 1
			else:
				data[i], data[k] = data[k], data[i]
				k -= 1
		return k

	@staticmethod
	def _insertion_sort(data: ['BVH._BVHPrimitive'], start: INT, end: INT, dim: INT):
		for i in range(start, end):
			k = i
			dk = data[k]
			while k > start and dk.centroid[dim] < data[k-1].centroid[dim]:
				data[k] = data[k-1]
				k -= 1
			data[k] = dk

	@staticmethod
	def _bucket_partition(data: ['BVH._BVHPrimitive'], start: INT, end: INT,
	                      min_cost_idx: INT, n_buckets: INT, dim: INT, centroid_bounds: [geo.BBox]):
		i = start
		k = end - 1
		diff = centroid_bounds.pMax[dim] - centroid_bounds.pMin[dim]
		while i < k:
			b = n_buckets * ((data[i].centroid[dim] - centroid_bounds.pMin[dim]) / diff)
			if b == n_buckets:
				b -= 1

			if b <= min_cost_idx:
				i += 1
			else:
				data[i], data[k] = data[k], data[i]
				k -= 1
		return k

	def _recursive_build(self, data: ['BVH._BVHPrimitive'], segment: [UINT], ordered_prims: ['BVH._BVHPrimitive']):
		"""Recursively build BVH, segment: [start, end, total]"""
		from pytracer.aggregate.accelerator.bvh import BVH
		start, end = segment[0:2]
		segment[2] += 1
		node = BVH._BVHNode()

		# compute bounds of primitives
		bbox = geo.BBox()
		for i in range(start, end):
			bbox.union(data[i].bounds)

		n_prim = end - start
		if n_prim == 1:
			# leaf node
			first = len(ordered_prims)
			for i in range(start, end):
				ordered_prims.append(self.primitives[data[i].prim_idx])

			node.init_leaf(first, n_prim, bbox)
		else:
			# compute centroids, choose split dimension
			mid = (start + end) // 2
			centroid_bounds = geo.BBox()
			for i in range(start, end):
				centroid_bounds.union(data[i].centroid)
			dim = centroid_bounds.maximum_extent()

			# partition prims into two sets, build children
			if centroid_bounds.pMax[dim] == centroid_bounds.pMin[dim]:
				if n_prim <= self.max_prim_per_node:
					# create leaf node
					first = len(ordered_prims)
					for i in range(start, end):
						ordered_prims.append(self.primitives[data[i]].prim_idx)
					node.init_leaf(first, n_prim, bbox)
				else:
					segment[1] = mid
					c0 = self._recursive_build(data, segment, ordered_prims)
					segment[0:2] = [mid, end]
					c1 = self._recursive_build(data, segment, ordered_prims)
					node.init_inter(dim, c0, c1)

				return node

			# partition based on split_method
			if self.split_method == BVH.SplitMethod.MIDDLE:
				pmid = .5 * (centroid_bounds.pMax[dim] + centroid_bounds.pMax[dim])
				mid = BVH._partition(data, start, end, pmid)

			else:
				# surface area heuristic
				if n_prim <= 4:
					# partition into equally-sized subsets
					mid = (start + end) // 2
					BVH._insertion_sort(data, start, end, dim)
				else:
					# buckets
					n_buckets = 12
					buckets = [[0, geo.BBox()] for i in range(n_buckets)]  # bucket: [count: INT, bounds: BBox]
					for i in range(start, end):
						b = INT(n_buckets * ((data[i].centroid[dim] - centroid_bounds.pMax[dim]) /
						                     (centroid_bounds.pMax[dim] - centroid_bounds.pMin[dim])))

						if b == n_buckets:
							b -= 1
						buckets[b][0] += 1
						buckets[b][1].union(data[i].bounds)

					# compute costs
					costs = [0.] * (n_buckets - 1)
					for i in range(n_buckets - 1):
						b0 = geo.BBox()
						b1 = geo.BBox()
						c0, c1 = 0, 0
						for j in range(0, i + 1):
							b0.union(buckets[j][1])
							c0 += buckets[j][0]
						for j in range(i + 1, n_buckets):
							b1.union(buckets[j][1])
							c1 += buckets[j][0]
						# cost for intersection: 1.
						# cost for traversal: .125
						costs[i] = .125 + (c0 * b0.surface_area() + b1.surface_area()) /\
						                  bbox.surface_area()

					# find bucket
					min_cost_idx = np.argmin(costs)

					# create leaf of split at bucket
					if n_prim > self.max_prim_per_node or costs[min_cost_idx] < n_prim:
						pmid = BVH._bucket_partition(data, start, end, min_cost_idx, n_buckets, dim, centroid_bounds)
						mid = pmid - start
					else:
						first = len(ordered_prims)
						for i in range(start, end):
							ordered_prims.append(self.primitives[data[i]].prim_idx)
						node.init_leaf(first, n_prim, bbox)
						return node

			segment[1] = mid
			c0 = self._recursive_build(data, segment, ordered_prims)
			segment[0:2] = [mid, end]
			c1 = self._recursive_build(data, segment, ordered_prims)
			node.init_inter(dim, c0, c1)

		return node
	
	def _flatten_tree(self, node: 'BVH._BVHNode', offset: UINT):
		# pre-traversal
		linear_node = self.nodes[offset]
		linear_node.bounds = node.bounds
		if node.n_prim > 0:
			linear_node.offset = node.first_offset
			linear_node.n_prim = node.n_prim
		else:
			linear_node.axis = node.split_axis
			linear_node.n_prim = 0
			off = self._flatten_tree(node.children[0], offset + 1)
			linear_node.offset = self._flatten_tree(node.children[1], off)
		
		return offset + 1
	
	@staticmethod
	def _intersect_p(bounds: geo.BBox, ray :geo.Ray, inv_dir: geo.Vector, dir_neg: [bool]) -> bool:
		# ray intersection against x and y slabs
		tmin = (bounds[dir_neg[0]].x - ray.o.x) * inv_dir.x
		tmax = (bounds[1-dir_neg[0]].x - ray.o.x) * inv_dir.x
		tmin_y = (bounds[dir_neg[0]].y - ray.o.y) * inv_dir.y
		tmax_y = (bounds[1 - dir_neg[0]].y - ray.o.y) * inv_dir.y
		
		if tmin > tmax_y or tmin_y > tmax:
			return False
		if tmin_y < tmin:
			tmin = tmin_y
		if tmax_y > tmax:
			tmax = tmax_y
		# z slab
		tmin_z = (bounds[dir_neg[0]].z - ray.o.z) * inv_dir.z
		tmax_z = (bounds[1 - dir_neg[0]].z - ray.o.z) * inv_dir.z

		if tmin > tmax_z or tmin_z > tmax:
			return False
		if tmin_z < tmin:
			tmin = tmin_z
		if tmax_z > tmax:
			tmax = tmax_z
	
		return tmin < ray.maxt and tmax > ray.mint

	def intersect(self, ray: 'geo.Ray', isect: 'Intersection') -> bool:
		from pytracer.aggregate.accelerator.bvh import BVH
		if self.nodes is None:
			return False
		hit = False
		inv_dir = 1. / ray.d
		dir_neg = ray.d < 0.
		
		todo_idx = 0
		node_idx = 0
		todo = [None] * 64
		while True:
			node = self.nodes[node_idx]
			# check intersection
			if BVH._intersect_p(node.bounds, ray, inv_dir, dir_neg):
				if node.n_prim > 0:
					# intersect with primitives in the leaf
					for i in range(node.n_prim):
						if self.primitives[node.offset + i].intersect(ray, isect):
							hit = True
					if todo_idx == 0:
						break
					todo_idx -= 1
					node_idx = todo[todo_idx]
				
				else:
					# advance to near node
					if dir_neg[node.axis]:
						todo[todo_idx] = node_idx + 1
						todo_idx += 1
						node_idx = node.offset

					else:
						todo[todo_idx] = node.offset
						todo_idx += 1
						node_idx += 1
			
			else:
				if todo_idx == 0:
					break
				todo_idx -= 1
				node_idx = todo[todo_idx]
		
		return hit

	def intersect_p(self, ray :'geo.Ray') -> bool:
		from pytracer.aggregate.accelerator.bvh import BVH
		if self.nodes is None:
			return False
		origin = ray(ray.mint)
		inv_dir = 1. / ray.d
		dir_neg = ray.d < 0.

		todo_idx = 0
		node_idx = 0
		todo = [None] * 64
		while True:
			node = self.nodes[node_idx]
			# check intersection
			if BVH._intersect_p(node.bounds, ray, inv_dir, dir_neg):
				if node.n_prim > 0:
					# intersect with primitives in the leaf
					for i in range(node.n_prim):
						if self.primitives[node.offset + i].intersect_p(ray):
							return True
					if todo_idx == 0:
						break
					todo_idx -= 1
					node_idx = todo[todo_idx]

				else:
					# advance to near node
					if dir_neg[node.axis]:
						todo[todo_idx] = node_idx + 1
						todo_idx += 1
						node_idx = node.offset

					else:
						todo[todo_idx] = node.offset
						todo_idx += 1
						node_idx += 1

			else:
				if todo_idx == 0:
					break
				todo_idx -= 1
				node_idx = todo[todo_idx]

		return False

	def world_bound(self):
		if self.nodes is not None:
			return self.nodes[0].bounds
		return geo.BBox()

	def can_intersect(self) -> bool:
		return True

	def refine(self, refined: ['Primitive']):
		raise NotImplementedError('{}.refine(): Not implemented'.format(self.__class__))
