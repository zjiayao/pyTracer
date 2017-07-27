'''
transform.py

This module is part of the pyTracer, which
defines the geometric transformations.

v0.0
Created by Jiayao on July 27, 2017
'''
'''
imp.reload(src.core.pytracer)
imp.reload(src.core.geometry)
imp.reload(src.core.transform)
from src.core.pytracer import *
from src.core.geometry import *
from src.core.transform import *
'''

from src.core.pytracer import *
from src.core.geometry import *
import numpy as np


class Transform():
	'''
	Transform class
	'''
	def __init__(self, m=None, mInv=None, dtype=FLOAT):
		if m is None:
			self.__m = np.eye(4,4, dtype=dtype)
			self.__mInv = np.eye(4,4, dtype=dtype)
		elif not mInv is None:
			self.__m = m.copy()
			self.__mInv = mInv.copy()
		else:
			if not np.shape(m) == (4,4):
				raise TypeError('Transform matrix must be 4x4')
			self.__m = m.copy()
			self.__mInv = np.linalg.inv(m)
		self.__m.flags.writeable = False
		self.__mInv.flags.writeable = False

	@property
	def m(self):
		return self.__m

	@property
	def mInv(self):
		return self.__mInv
	
	def __repr__(self):
		return "{}\nTransformation:\n{}\nInverse Transformation:\n{}" \
			.format(self.__class__, self.m, self.mInv)

	def __eq__(self, other):
		return np.array_equal(self.m, other.m)

	def __ne__(self, other):
		return not np.array_equal(self.m, other.m)

	def __call__(self, arg):
		if isinstance(arg, Point):
			res = self.m[0:4, 0:3].dot(arg) + self.m[0:4, 3]
			p = Point(res[0], res[1], res[2])
			if ne_unity(res[3]):
				p /= res[3]
			return p
		
		elif isinstance(arg, Vector):
			res = self.m[0:3, 0:3].dot(arg)
			return Vector(res[0], res[1], res[2])

		elif isinstance(arg, Normal):
			# must be transformed by inverse transpose
			res = self.mInv[0:3, 0:3].T.dot(arg)
			return Normal(res[0], res[1], res[2])

		elif isinstance(arg, Ray):
			r = Ray.fromRay(arg)
			r.o = self(r.o)
			r.d = self(r.d)
			return r

		elif isinstance(arg, BBox):
			res = BBox.fromBBox(arg)
			x = Vector(res.pMax.x - res.pMin.x, 0., 0.)
			y = Vector(res.pMax.y - res.pMin.y, 0., 0.)
			z = Vector(res.pMax.z - res.pMin.z, 0., 0.)
			res.pMin = self(res.pMin)
			x = self(x)
			y = self(y)
			z = self(z)
			res.pMax = res.pMin + (x + y + z)
			return res

		else:
			raise TypeError('Transform can only be called on Point or Vector')

	def __mul__(self, other):
		m = self.m.dot(other.m)
		mInv = other.mInv.dot(self.mInv)
		return Transform(m, mInv)

	@staticmethod
	def translate(delta: Vector, dtype=FLOAT):
		m = np.eye(4,4, dtype=dtype)
		mInv = np.eye(4,4, dtype=dtype)

		m[0][3] = delta.x
		m[1][3] = delta.y
		m[2][3] = delta.z
		mInv[0][3] = -delta.x
		mInv[1][3] = -delta.y
		mInv[2][3] = -delta.z

		return Transform(m, mInv, dtype)

	@staticmethod
	def scale(x, y, z, dtype=FLOAT):
		m = np.eye(4,4, dtype=dtype)
		mInv = np.eye(4,4, dtype=dtype)	

		m[0][0] = x
		m[1][1] = y
		m[2][2] = z
		mInv[0][0] = 1. / x
		mInv[1][1] = 1. / y
		mInv[2][2] = 1. / z

		return Transform(m, mInv, dtype)

	# all angles are in degrees
	@staticmethod
	def rotate_x(angle, dtype=FLOAT):
		m = np.eye(4, 4, dtype=dtype)
		sin_t = np.sin(np.deg2rad(angle))
		cos_t = np.cos(np.deg2rad(angle))
		m[1][1] = cos_t
		m[1][2] = -sin_t
		m[2][1] = sin_t
		m[2][2] = cos_t
		return Transform(m, m.T, dtype)

	@staticmethod
	def rotate_y(angle, dtype=FLOAT):
		m = np.eye(4, 4, dtype=dtype)
		sin_t = np.sin(np.deg2rad(angle))
		cos_t = np.cos(np.deg2rad(angle))
		m[0][0] = cos_t
		m[0][2] = sin_t
		m[2][0] = -sin_t
		m[2][2] = cos_t
		return Transform(m, m.T, dtype)

	@staticmethod
	def rotate_z(angle, dtype=FLOAT):
		m = np.eye(4, 4, dtype=dtype)
		sin_t = np.sin(np.deg2rad(angle))
		cos_t = np.cos(np.deg2rad(angle))
		m[0][0] = cos_t
		m[0][1] = -sin_t
		m[1][0] = sin_t
		m[1][1] = cos_t
		return Transform(m, m.T, dtype)

	@staticmethod
	def rotate(angle, axis:Vector, dtype=FLOAT):
		a = normalize(axis)

		s = np.sin(np.deg2rad(angle))
		c = np.cos(np.deg2rad(angle))

		m = np.eye(4, 4, dtype=dtype)

		m[0][0] = a.x * a.x + (1. - a.x * a.x) * c
		m[0][1] = a.x * a.y * (1. - c) - a.z * s
		m[0][2] = a.x * a.z * (1. - c) + a.y * s
		m[1][0] = a.x * a.y * (1. - c) + a.z * s
		m[1][1] = a.y * a.y + (1. - a.y * a.y) * c
		m[1][2] = a.y * a.z * (1. - c) - a.x * s
		m[2][0] = a.x * a.z * (1. - c) - a.y * s
		m[2][1] = a.y * a.z * (1. - c) + a.x * s
		m[2][2] = a.z * a.z + (1. - a.z * a.z) * c

		return Transform(m, m.T, dtype)

	@staticmethod
	def look_at(pos: Point, look: Point, up: Vector, dtype=FLOAT):
		'''
		look_at
		Look-at transformation, from camera
		to world
		'''
		w2c = np.eye(4, 4, dtype=dtype)
		c2w = np.eye(4, 4, dtype=dtype)

		zc = normalize(look - pos)
		xc = normalize( normalize(up).cross(zc) )
		yc = zc.cross(xc) # orthogonality		

		# c2w translation
		c2w[0][3] = pos.x
		c2w[1][3] = pos.y
		c2w[2][3] = pos.z

		# c2w rotation
		c2w[0][0] = xc.x
		c2w[0][1] = xc.y
		c2w[0][2] = xc.z
		c2w[1][0] = yc.x
		c2w[1][1] = yc.y
		c2w[1][2] = yc.z
		c2w[2][0] = zc.x
		c2w[2][1] = zc.y
		c2w[2][2] = zc.z

		# w2c rotation
		# in effect as camera extrinsic
		w2c[0][0] = xc.x
		w2c[0][1] = yc.x
		w2c[0][2] = zc.x
		w2c[1][0] = xc.y
		w2c[1][1] = yc.y
		w2c[1][2] = zc.y
		w2c[2][0] = xc.z
		w2c[2][1] = yc.z
		w2c[2][2] = zc.z		

		# w2c translation
		w2c[0][3] = -(pos.x * xc.x + pos.y * yc.x + pos.z * zc.x)
		w2c[1][3] = -(pos.x * xc.y + pos.y * yc.y + pos.z * zc.y)
		w2c[2][3] = -(pos.x * xc.z + pos.y * yc.z + pos.z * zc.z)

		return Transform(c2w, w2c, dtype)

	def inverse(self):
		'''
		Returns the inverse transformation
		'''
		return Transform(self.mInv, self.m)

	def is_identity(self):
		return np.array_equal(self.m, np.eye(4,4))

	def has_scale(self):
		return ne_unity(self.m[0][0] * self.m[0][0]) or \
			   ne_unity(self.m[1][1] * self.m[1][1]) or \
			   ne_unity(self.m[2][2] * self.m[2][2])

	def swaps_handness(self):
		return np.linalg.det(self.m[0:3,0:3]) < 0.





		



