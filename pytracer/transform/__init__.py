"""
__init__.py

pytracer.transform package

Created by Jiayao on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo


class Transform(object):
	"""
	Transform class
	"""
	def __init__(self, m=None, m_inv=None, dtype=FLOAT):
		if m is None:
			self.__m = np.eye(4, 4, dtype=dtype)
			self.__m_inv = np.eye(4, 4, dtype=dtype)
		elif m is not None and m_inv is not None:
			self.__m = m.copy()
			self.__m_inv = m_inv.copy()
		else:
			if not np.shape(m) == (4, 4):
				raise TypeError('Transform matrix must be 4x4')

			self.__m = m.copy()
			self.__m_inv = np.linalg.inv(m)
		self.__m.flags.writeable = False
		self.__m_inv.flags.writeable = False

	@property
	def m(self):
		return self.__m

	@property
	def m_inv(self):
		return self.__m_inv

	def __repr__(self):
		return "{}\nTransformation:\n{}\nInverse Transformation:\n{}" \
			.format(self.__class__, self.m, self.m_inv)

	def __eq__(self, other):
		return np.array_equal(self.m, other.m)

	def __ne__(self, other):
		return not np.array_equal(self.m, other.m)

	def copy(self):
		return Transform(self.m.copy(), self.m_inv.copy())

	def __call__(self, arg, dtype=None):
		if isinstance(arg, geo.Point) or (isinstance(arg, np.ndarray) and dtype == geo.Point):
			res = self.m[0:4, 0:3].dot(arg) + self.m[0:4, 3]
			p = geo.Point(res[0], res[1], res[2])
			if util.ne_unity(res[3]):
				p /= res[3]
			return p

		elif isinstance(arg, geo.Vector) or (isinstance(arg, np.ndarray) and dtype == geo.Vector):
			res = self.m[0:3, 0:3].dot(arg)
			return geo.Vector(res[0], res[1], res[2])

		elif isinstance(arg, geo.Normal) or (isinstance(arg, np.ndarray) and dtype == geo.Normal):
			# must be transformed by inverse transpose
			res = self.m_inv[0:3, 0:3].T.dot(arg)
			return geo.Normal(res[0], res[1], res[2])

		elif isinstance(arg, geo.RayDifferential):
			r = geo.RayDifferential.from_rd(arg)
			r.o = self(r.o)
			r.d = self(r.d)
			return r

		elif isinstance(arg, geo.Ray):
			r = geo.Ray.from_ray(arg)
			r.o = self(r.o)
			r.d = self(r.d)
			return r

		elif isinstance(arg, geo.BBox):
			res = geo.BBox.from_bbox(arg)
			x = geo.Vector(res.pMax.x - res.pMin.x, 0., 0.)
			y = geo.Vector(0., res.pMax.y - res.pMin.y, 0.)
			z = geo.Vector(0., 0., res.pMax.z - res.pMin.z)
			res.pMin = self(res.pMin)
			x = self(x)
			y = self(y)
			z = self(z)
			res.pMax = res.pMin + (x + y + z)
			return res

		else:
			raise TypeError('Transform can only be called on geo.Point, geo.Vector, geo.Normal, geo.Ray or geo.geo.BBox')

	def __mul__(self, other):
		m = self.m.dot(other.m)
		m_inv = other.m_inv.dot(self.m_inv)
		return Transform(m, m_inv)

	@classmethod
	def translate(cls, delta: 'geo.Vector', dtype=FLOAT) -> 'Transform':
		m = np.eye(4, 4, dtype=dtype)
		m_inv = np.eye(4, 4, dtype=dtype)

		m[0][3] = delta.x
		m[1][3] = delta.y
		m[2][3] = delta.z
		m_inv[0][3] = -delta.x
		m_inv[1][3] = -delta.y
		m_inv[2][3] = -delta.z

		return cls(m, m_inv, dtype)

	@classmethod
	def scale(cls, x, y, z, dtype=FLOAT) -> 'Transform':
		m = np.eye(4, 4, dtype=dtype)
		m_inv = np.eye(4, 4, dtype=dtype)

		m[0][0] = x
		m[1][1] = y
		m[2][2] = z
		m_inv[0][0] = 1. / x
		m_inv[1][1] = 1. / y
		m_inv[2][2] = 1. / z

		return cls(m, m_inv, dtype)

	# all angles are in degrees
	@classmethod
	def rotate_x(cls, angle, dtype=FLOAT) -> 'Transform':
		m = np.eye(4, 4, dtype=dtype)
		sin_t = np.sin(np.deg2rad(angle))
		cos_t = np.cos(np.deg2rad(angle))
		m[1][1] = cos_t
		m[1][2] = -sin_t
		m[2][1] = sin_t
		m[2][2] = cos_t
		return cls(m, m.T, dtype)

	@classmethod
	def rotate_y(cls, angle, dtype=FLOAT) -> 'Transform':
		m = np.eye(4, 4, dtype=dtype)
		sin_t = np.sin(np.deg2rad(angle))
		cos_t = np.cos(np.deg2rad(angle))
		m[0][0] = cos_t
		m[0][2] = sin_t
		m[2][0] = -sin_t
		m[2][2] = cos_t
		return cls(m, m.T, dtype)

	@classmethod
	def rotate_z(cls, angle, dtype=FLOAT) -> 'Transform':
		m = np.eye(4, 4, dtype=dtype)
		sin_t = np.sin(np.deg2rad(angle))
		cos_t = np.cos(np.deg2rad(angle))
		m[0][0] = cos_t
		m[0][1] = -sin_t
		m[1][0] = sin_t
		m[1][1] = cos_t
		return cls(m, m.T, dtype)

	@classmethod
	def rotate(cls, angle, axis: 'geo.Vector', dtype=FLOAT) -> 'Transform':
		a = geo.normalize(axis)

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

		return cls(m, m.T, dtype)

	@classmethod
	def look_at(cls, pos: 'geo.Point', look: 'geo.Point', up: 'geo.Vector', dtype=FLOAT) -> 'Transform':
		"""
		look_at
		Look-at transformation, from camera
		to world
		"""
		w2c = np.eye(4, 4, dtype=dtype)
		c2w = np.eye(4, 4, dtype=dtype)

		zc = geo.normalize(look - pos)
		xc = geo.normalize(geo.normalize(up).cross(zc))
		yc = zc.cross(xc)  # orthogonality

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

		return cls(c2w, w2c, dtype)

	@classmethod
	def orthographic(cls, znear: FLOAT, zfar: FLOAT):
		return cls.scale(1., 1., 1. / (zfar - znear)) * cls.translate(geo.Vector(0., 0., -znear))

	@classmethod
	def perspective(cls, fov: FLOAT, n: FLOAT, f: FLOAT, dtype=FLOAT):
		# projective along z
		m = np.eye(4, dtype=dtype)
		m[2, 2] = f / (f - n)
		m[2, 3] = -f * n / (f - n)
		m[3, 2] = 1.
		m[3, 3] = 0.

		# scale to viewing volume
		tan_inv = 1. / np.tan(np.deg2rad(fov) / 2.)
		return cls.scale(tan_inv, tan_inv, 1.) * cls(m)

	def inverse(self) -> 'Transform':
		"""
		Returns the inverse transformation
		"""
		return Transform(self.m_inv, self.m)

	def is_identity(self) -> bool:
		return np.array_equal(self.m, np.eye(4, 4))

	def has_scale(self) -> bool:
		return util.ne_unity(self.m[0][0:3].dot(self.m[0][0:3])) or \
		       util.ne_unity(self.m[1][0:3].dot(self.m[1][0:3])) or \
		       util.ne_unity(self.m[2][0:3].dot(self.m[2][0:3]))

	def swaps_handedness(self) -> bool:
		return (np.linalg.det(self.m[0:3, 0:3]) < 0.)


class AnimatedTransform(object):
	def __init__(self, t1: 'Transform', tm1: FLOAT, t2: 'Transform', tm2: FLOAT):
		self.startTime = tm1
		self.endTime = tm2
		self.startTransform = t1
		self.endTransform = t2
		self.animated = (t1 != t2)
		self.T = [None, None]
		self.R = [None, None]
		self.S = [None, None]
		self.T[0], self.R[0], self.S[0] = AnimatedTransform.decompose(t1.m)
		self.T[1], self.R[1], self.S[1] = AnimatedTransform.decompose(t2.m)

	def __repr__(self):
		return "{}\nTime: {} - {}\nAnimated: {}".format(self.__class__,
		                                                self.startTime, self.endTime, self.animated)

	def __call__(self, arg_1, arg_2=None):
		if isinstance(arg_1, geo.Ray) and arg_2 is None:
			r = arg_1
			if not self.animated or r.time < self.startTime:
				tr = self.startTransform(r)
			elif r.time >= self.endTime:
				tr = self.endTransform(r)
			else:
				tr = self.interpolate(r.time)(r)
			tr.time = r.time
			return tr

		elif isinstance(arg_1, (float, FLOAT, np.float)) and isinstance(arg_2, geo.Point):
			time = arg_1
			p = arg_2
			if not self.animated or time < self.startTime:
				return self.startTransform(p)
			elif time > self.endTime:
				return self.endTransform(p)
			return self.interpolate(time)(p)

		elif isinstance(arg_1, (float, FLOAT, np.float)) and isinstance(arg_2, geo.Vector):
			time = arg_1
			v = arg_2
			if not self.animated or time < self.startTime:
				return self.startTransform(v)
			elif time > self.endTime:
				return self.endTransform(v)
			return self.interpolate(time)(v)
		else:
			raise TypeError()

	def motion_bounds(self, b: 'geo.BBox', use_inv: bool=False) -> 'geo.BBox':
		if not self.animated:
			return self.startTransform.inverse()(b)
		ret = geo.BBox()
		steps = 128
		for i in range(128):
			time = util.lerp(i * (1. / (steps - 1)), self.startTime, self.endTime)
			t = self.interpolate(time)
			if use_inv:
				t = t.inverse()
			ret.union(t(b))
		return ret

	@staticmethod
	def decompose(m: 'np.ndarray') -> ['geo.Vector', 'quat.Quaternion', 'np.ndarray']:
		"""
		decompose into
		m = T R S
		Assume m is an affine transformation
		"""
		if not np.shape(m) == (4, 4):
			raise TypeError

		T = geo.Vector(m[0, 3], m[1, 3], m[2, 3])
		M = m.copy()
		M[0:3, 3] = M[3, 0:3] = 0
		M[3, 3] = 1

		# polar decomposition
		norm = 2 * EPS
		R = M.copy()

		for _ in range(100):
			if norm < EPS:
				break
			Rit = np.linalg.inv(R.T)
			Rnext = .5 * (Rit + R)
			D = np.fabs(Rnext - Rit)[0:3, 0:3]
			norm = max(norm, np.max(np.sum(D, axis=0)))
			R = Rnext

		from pytracer.transform.quat import from_arr
		Rquat = from_arr(R)
		S = np.linalg.inv(R).dot(M)

		return T, Rquat, S

	def interpolate(self, time: FLOAT) -> 'Transform':

		if not self.animated or time <= self.startTime:
			return self.startTransform

		if time >= self.endTime:
			return self.endTransform

		from pytracer.transform.quat import (slerp, to_transform)

		dt = (time - self.startTime) / (self.endTime - self.startTime)

		trans = (1. - dt) * self.T[0] + dt * self.T[1]
		rot = slerp(dt, self.R[0], self.R[1])
		scale = util.ufunc_lerp(dt, self.S[0], self.S[1])

		return Transform.translate(trans) *\
		       to_transform(rot) *\
		       Transform(scale)
