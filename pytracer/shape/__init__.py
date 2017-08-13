"""
__init__.py

pytracer.shape package

Created by Jiayao on Aug 13, 2017
"""
from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
from .. import *
from .. import geometry as geo
from .. import transform as trans


# aux functions for mesh navigation


def NEXT(i: INT) -> INT:
	return (i + 1) % 3


def PREV(i: INT) -> INT:
	return (i + 2) % 3


class Shape(object, metaclass=ABCMeta):
	"""
	Shape Class

	Base class of shapes.
	"""

	next_shapeId = 1

	def __init__(self, o2w: 'trans.Transform', w2o: 'trans.Transform',
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
	def object_bound(self) -> 'geo.BBox':
		raise NotImplementedError('unimplemented Shape.object_bound() method called')

	def world_bound(self) -> 'geo.BBox':
		return self.o2w(self.object_bound())

	def can_intersect(self) -> bool:
		return True

	@abstractmethod
	def refine(self) -> ['Shape']:
		"""
		If `Shape` cannot intersect,
		return a refined subset
		"""
		raise NotImplementedError('Intersecable shapes are not refineable')

	@abstractmethod
	def intersect(self, r: 'geo.Ray') -> (bool, FLOAT, FLOAT, 'geo.DifferentialGeometry'):
		raise NotImplementedError('unimplemented Shape.intersect() method called')

	@abstractmethod
	def intersect_p(self, r: 'geo.Ray') -> bool:
		raise NotImplementedError('unimplemented {}.intersect_p() method called'
		                          .format(self.__class__))

	def get_shading_geometry(self, o2w: 'trans.Transform',
	                         dg: 'geo.DifferentialGeometry'):
		return dg.copy()

	@abstractmethod
	def area(self) -> FLOAT:
		raise NotImplementedError('unimplemented Shape.area() method called')

	def sample(self, u1: FLOAT, u2: FLOAT) -> ['geo.Point', 'geo.Normal']:
		"""
		sample()

		Returns the position and
		surface normal randomly chosen
		"""
		raise NotImplementedError('unimplemented {}.sample() method called'
		                          .format(self.__class__))

	def sample_p(self, pnt: 'geo.Point', u1: FLOAT, u2: FLOAT) -> ['geo.Point', 'geo.Normal']:
		"""
		sample_p()

		Returns the position and
		surface normal randomly chosen
		s.t. visible to `pnt`
		"""
		return self.sample(u1, u2)

	def pdf(self, pnt: 'geo.Point') -> FLOAT:
		"""
		pdf()

		Return the sampling pdf
		"""
		return 1. / self.area()

	def pdf_p(self, pnt: 'geo.Point', wi: 'geo.Vector') -> FLOAT:
		"""
		pdf_p()

		Return the sampling pdf w.r.t.
		the sample_p() method.
		Transforms the density from one
		defined over area to one defined
		over solid angle from `pnt`.
		"""
		# intersect sample ray with area light
		ray = geo.Ray(pnt, wi, EPS)

		hit, thit, r_eps, dg_light = self.intersect(ray)
		if not hit:
			return 0.

		# convert light sample weight to solid angle measure
		return (pnt - ray(thit)).sq_length() / (dg_light.nn.abs_dot(-wi) * self.area())
