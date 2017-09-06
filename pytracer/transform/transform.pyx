"""
transform.py

pytracer.transform package

Created by Jiayao on Aug 13, 2017
Cythonized on Aug 30, 2017
"""
# from pytracer.geometry import (Point, Vector, Normal, Ray, RayDifferential)
import numpy as np
cimport numpy as np
import cython

cdef mat4x4 mat4x4_inv(mat4x4 mat):
	# cdef np.ndarray view = np.asarray(mat)
	cdef mat4x4 ret = np.linalg.inv(mat).astype(np.dtype('float32'))
	# cdef mat4x4 ret = np.linalg.inv(mat).astype(np.dtype('float32'))
	return ret
	# cdef int k = 4, zero = 0
	# cdef DOUBLE_t *mat_ptr = <DOUBLE_t*> np.PyArray_DATA(mat)
	# cdef DOUBLE_t *id_ptr
	# cdef int *pvt_ptr = <int*> malloc(sizeof (int) * k)
	# identity = np.eye(k, dtype=np.dtype('float64'))
	# cdef mat4x4 buffer
	# try:
	# 	id_ptr = <DOUBLE_t*> np.PyArray_DATA(identity)
	# 	cython_lapack.dgesv(&k, &k, mat_ptr, &k,
	# 	                    pvt_ptr, id_ptr, &k, &zero)
	# 	buffer = identity.astype(np.dtype('float32'))
	# 	return buffer
	#
	# finally:
	# 	free(pvt_ptr)

cdef class Transform:
	"""
	Transform class
	"""
	@cython.boundscheck(False)
	def __cinit__(self, mat4x4 m=None, mat4x4 m_inv=None):
		if m is None:
			self.m = np.eye(4, dtype=np.dtype('float32'))
			self.m_inv = np.eye(4, dtype=np.dtype('float32'))
		elif m is not None and m_inv is not None:
			if m.shape[0] != 4 or m.shape[1] != 4 or m_inv.shape[0] != 4 or m_inv.shape[1] != 4:
				raise TypeError
			self.m = np.empty([4, 4], dtype=np.dtype('float32'))
			self.m_inv = np.empty([4, 4], dtype=np.dtype('float32'))
			self.m[:, :] = m
			self.m_inv[:, :] = m_inv
		else:
			if m.shape[0] != 4 or m.shape[1] != 4:
				raise TypeError
			self.m = np.empty([4, 4], dtype=np.dtype('float32'))
			self.m_inv = np.empty([4, 4], dtype=np.dtype('float32'))
			self.m[:, :] = m
			self.m_inv = mat4x4_inv(m)

	@property
	def m(self):
		return self.m

	@property
	def m_inv(self):
		return self.m_inv

	def __repr__(self):
		return "{}\nTransformation:\n{}\nInverse Transformation:\n{}" \
			.format(self.__class__, self.m, self.m_inv)

	def __richcmp__(x, Transform y, INT_t op):
		cdef:
			INT_t i, j

		assert isinstance(x, Transform) and isinstance(y, Transform)

		if op == Py_EQ:
			for i in range(4):
				for j in range(4):
					if x.m[i][j] != y.m[i][j]:
						return False
					if x.m_inv[i][j] != y.m_inv[i][j]:
						return False
			return True

		elif op == Py_NE:
			for i in range(4):
				for j in range(4):
					if x.m[i][j] == y.m[i][j]:
						return False
					if x.m_inv[i][j] == y.m_inv[i][j]:
						return False
			return True

		raise NotImplementedError


	cpdef Transform copy(self):
		cdef Transform to = Transform()
		self._copy(to)
		return to


	def __call__(self, arg):
		print("Depreciated, using c methods instead")
		cdef:
			_Arr3 ret
			Ray ray
			RayDifferential rd
			BBox box

		if isinstance(arg, Point):
			arg = <Point> arg
			ret = Point()
			self._call_point(arg, ret)

			return ret

		elif isinstance(arg, Vector):
			arg = <Vector> arg
			ret = Vector()
			self._call_vector(arg, ret)
			return ret

		elif isinstance(arg, Normal):
			# must be transformed by inverse transpose
			arg = <Normal> arg
			ret = Normal()
			self._call_normal(arg, ret)
			return ret

		elif isinstance(arg, RayDifferential):
			rd = RayDifferential()
			arg = <RayDifferential> arg
			self._call_ray(arg, rd)
			assert isinstance(rd, RayDifferential)
			return rd

		elif isinstance(arg, Ray):
			arg = <Ray> arg
			ray = Ray()
			self._call_ray(arg, ray)
			return ray

		elif isinstance(arg, BBox):
			arg = <BBox> arg
			box = BBox()
			self._call_bbox(arg, box)
			return box

		else:
			raise TypeError('Transform can only be called on geo.Point, geo.Vector, geo.Normal, geo.Ray or geo.geo.BBox')

	def __mul__(self, Transform other):
		cdef Transform ret = Transform()
		assert isinstance(other, Transform)
		mat4x4_kji(self.m, other.m, ret.m)
		mat4x4_kji(other.m_inv, self.m_int, ret.m_inv)
		return ret

	@staticmethod
	def translate(Vector delta):
		cdef Transform trans = Transform()
		Transform._translate(trans, delta)
		return trans

	@staticmethod
	def scale(FLOAT_t x, FLOAT_t y, FLOAT_t z):
		cdef Transform trans = Transform()
		Transform._scale(trans, x, y, z)
		return trans

	# all angles are in degrees
	@staticmethod
	def rotate_x(FLOAT_t angle):
		cdef Transform ret = Transform()
		Transform._rotate_x(ret, angle)
		return ret

	@staticmethod
	def rotate_y(FLOAT_t angle):
		cdef Transform ret = Transform()
		Transform._rotate_y(ret, angle)
		return ret

	@staticmethod
	def rotate_z(FLOAT_t angle):
		cdef Transform ret = Transform()
		Transform._rotate_z(ret, angle)
		return ret

	@staticmethod
	def rotate(FLOAT_t angle, Vector axis):
		cdef Transform ret = Transform()
		Transform._rotate(ret, angle, axis)
		return ret

	@staticmethod
	cdef void _look_at(Transform trans, Point pos, Point look, Vector up):
		cdef:
			Vector zc = _normalize(look - pos)
			Vector xc = _normalize(_normalize(up)._cross(zc))
			Vector yc = zc._cross(xc)
			FLOAT_t[:, :] w2c = trans.m_inv
			FLOAT_t[:, :] c2w = trans.m


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

	@staticmethod
	def look_at(Point pos, Point look, Vector up):
		cdef Transform ret = Transform()
		Transform._look_at(ret, pos, look, up)
		return ret

	@classmethod
	def orthographic(cls, FLOAT_t znear, FLOAT_t zfar):
		return cls.scale(1., 1., 1. / (zfar - znear)) * cls.translate(Vector(0., 0., -znear))

	@classmethod
	def perspective(cls, FLOAT_t fov, FLOAT_t n, FLOAT_t f):
		# projective along z
		cdef mat4x4 m = np.eye(4)
		cdef FLOAT_t tan_inv = 1. / ftan(deg2rad(fov) / 2.)

		m[2][2] = f / (f - n)
		m[2][3] = -f * n / (f - n)
		m[3][2] = 1.
		m[3][3] = 0.

		# scale to viewing volume
		return cls.scale(tan_inv, tan_inv, 1.) * cls(m)

	cpdef Transform inverse(self):
		return self._inverse()

	cpdef bint is_identity(self):
		return self._is_identity()

	cpdef bint has_scale(self):
		return self._has_scale()

	cpdef bint swaps_handedness(self):
		return self._swaps_handedness()
