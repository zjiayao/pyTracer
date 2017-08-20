"""
utility.py

pytracer.light.utility package

Utility classes and functions for Lights

Created by Jiayao on Aug 8, 2017
"""

from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo
import pytracer.sampler as spler
import pytracer.scene as scn
import pytracer.renderer as ren
import pytracer.montecarlo as mc

__all__ = ['VisibilityTester', 'ShapeSet',
           'LightSampleOffset', 'FLOAT', 'LightSample']


# Utility Classes
class VisibilityTester(object):
	"""
	VisibilityTester Class
	"""
	def __init__(self, ray: 'geo.Ray'=None):
		self.ray = None

	def __repr__(self):
		return "{}\ngeo.Ray: {}\n".format(self.__class__, self.ray)

	def set_segment(self, p1: 'geo.Point', eps1: FLOAT, p2: 'geo.Point', eps2: FLOAT, time: FLOAT):
		"""
		set_segment()

		The test is to be done within the
		given segment.
		"""
		dist = (p1 - p2).length()
		self.ray = geo.Ray(p1, (p2 - p1) / dist, eps1, dist * (1. - eps2), time)

	def set_ray(self, p: 'geo.Point', eps: FLOAT, w: 'geo.Vector', time: FLOAT):
		"""
		set_ray()

		The test is to indicate whether there
		is any object along a given direction.
		"""
		self.ray = geo.Ray(p, w, eps, np.inf, time)

	def unoccluded(self, scene: 'scn.Scene') -> bool:
		"""
		unoccluded()

		Traces a shadow ray
		"""
		return not scene.intersect_p(self.ray)

	def transmittance(self, scene: 'scn.Scene', renderer: 'ren.Renderer', sample: 'spler.Sample', rng=np.random.rand):
		"""
		transmittance()

		Determines the fraction of illumincation from
		the light to the point that is not extinguished
		by participating media.
		"""
		return renderer.transmittance(scene, geo.RayDifferential.from_ray(self.ray), sample, rng)


class ShapeSet(object):
	"""
	ShapeSet Class

	Wrapper for a set of `Shape`s.
	"""
	def __init__(self, shape: 'Shape'):
		self.shapes = []
		tmp = [shape]
		while len(tmp) > 0:
			sh = tmp.pop()
			if sh.can_intersect():
				self.shapes.append(sh)
			else:
				tmp.extend(sh.refine())

		if len(self.shapes) > 64:
			print("Warning: src.core.light.{}.__init__(): "
					"Area light turned into {} shapes, might be inefficient".format(self.__class__,
							len(self.shapes)))
		self.areas = []
		self.sum_area = 0.
		for sh in self.shapes:
			area = sh.area()
			self.areas.append(area)
			self.sum_area += area

		self.area_dist = mc.Distribution1D(self.areas)

	def __repr__(self):
		return "{}\nNumber of Shapes: {}\n".format(self.__class__, len(self.shapes))

	def sample_p(self, p: 'geo.Point', ls: 'LightSample') -> ['geo.Point', 'geo.Normal']:
		sn, _ = self.area_dist.sample_dis(ls.u_com)
		pt, ns = self.shapes[sn].sample_p(p, ls.u_pos[0], ls.u_pos[1])

		return [pt, ns]
		# find cloest intersection
		# r = geo.Ray(p, pt - p, EPS, np.inf)
		# any_hit = False

		# # inefficient
		# for sh in self.shapes:
		# 	hit, thit, _, dg = sh.intersect(r)
		# 	any_hit |= hit
		# if any_hit:
		# 	return [r(thit), dg.nn]
		# else:
		# 	return [pt, ns]

	def sample(self, ls: 'LightSample') -> ['geo.Point', 'geo.Normal']:
		sn, _ = self.area_dist.sample_dis(ls.u_com)
		pt, ns = self.shapes[sn].sample(ls.u_pos[0], ls.u_pos[1])

		return [pt, ns]

	def pdf(self, p: 'geo.Point', wi: 'geo.Vector') -> FLOAT:
		pdf = 0.
		for i, sh in enumerate(self.shapes):
			pdf += self.areas[i] * sh.pdf_p(p, wi)
		return pdf / self.sum_area

	def pdf_p(self, p: 'geo.Point') -> FLOAT:
		pdf = 0.
		for i, sh in enumerate(self.shapes):
			pdf += self.areas[i] * sh.pdf(p)
		return pdf / (self.sum_area * len(self.shapes))


class LightSampleOffset(object):
	"""
	LightSampleOffset Class

	Encapsulate offsets provided
	by sampler
	"""
	def __init__(self, nSamples: INT, sample: 'spler.Sample'):
		self.nSamples = nSamples
		self.offset_com = sample.add_1d(nSamples)
		self.offset_pos = sample.add_2d(nSamples)

	def __repr__(self):
		return "{}\nSamples: {}\n".format(self.__class__, self.nSamples)


class LightSample(object):
	"""
	LightSample Class

	Encapsulate Light samples,
	i.e., three dimensional random samples.
	"""
	def __init__(self, p0: FLOAT=0., p1: FLOAT=0., u_com: FLOAT=0.):
		self.u_pos = [p0, p1]
		self.u_com = u_com

	def __repr__(self):
		return "{}\nDirection: ({},{})\nComponent: {}\n".format(self.__class__, self.u_pos[0], self.u_pos[1], self.u_com)

	@classmethod
	def from_rand(cls, rng=np.random.rand):
		self = cls(rng(), rng(), rng())
		return self

	@classmethod
	def from_sample(cls, sample: 'spler.Sample', offset: 'LightSampleOffset', n: UINT):
		self = cls(sample.twoD[offset.offset_pos][n][0],
				   sample.twoD[offset.offset_pos][n][1],
				   sample.oneD[offset.offset_com][n])
		return self
