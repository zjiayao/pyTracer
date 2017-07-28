'''
diffgeo.py

This module is part of the pyTracer, which
defines differential geometric operations.

v0.0
Created by Jiayao on July 28, 2017
'''
'''
imp.reload(src.core.pytracer)
imp.reload(src.core.geometry)
imp.reload(src.core.transform)
from src.core.pytracer import *
from src.core.geometry import *
from src.core.transform import *
'''

import numpy as np
from src.core.pytracer import *
from src.core.geometry import *
from src.core.shape import *

class DifferentialGeometry:
	'''
	Differential Geometry class
	'''
	def __init__(self, P: 'Point', DPDU: 'Vector', DPDV: 'Vector',
				 DNDU: 'Vector', DNDV: 'Vector', uu: 'FLOAT',
				 vv: 'FLOAT', sh: 'Shape'):
		self.p = P.copy()
		self.dpdu = DPDU.copy()
		self.dpdv = DPDV.copy()
		self.dndu = DNDU.copy()
		self.dndv = DNDV.copy()

		self.nn = Normal.fromVector(normalize(DPDU.cross(DPDV)))
		self.u = uu
		self.v = vv
		self.shape = sh

		# adjust for handedness
		if sh is not None and \
			(sh.ro ^ sh.transform_swaps_handedness):
			self.nn = -1. * self.nn

	def __repr__(self):
		return "{}\nNormal Vector: {}".format(self.__class__, self.nn)

	def copy(self):
		return DifferentialGeometry(self.p, self.dpdu, self.dpdv,
				self.dndu, self.dndv, self.u, self.v, self.shape)












