"""
__init__.py

pytracer.geometry package

Created by Jiayao on Aug 13, 2017
"""
from __future__ import absolute_import
import numpy as np
cimport numpy as np
from typing import overload

FLOAT = np.float32
ctypedef np.float32_t FLOAT_t
# Classes


class Vector(np.ndarray):
	"""
	Vector Class

	A wrappper subclasses numpy.ndarray which
	models a 3D vector.
	"""
	def __new__(cls, x=0., y=0., z=0):
		return np.empty(3).view(cls)

	def __init__(self, x=0., y=0., z=0.):
		if np.isnan(x) or np.isnan(y) or np.isnan(z):
			raise ValueError
		super().__init__()
		self[0] = x
		self[1] = y
		self[2] = z

	@classmethod
	def from_arr(cls, n: 'np.ndarray'):  # Forward type hint (PEP-484)
		assert np.shape(n)[0] == 3
		return cls(n[0], n[1], n[2])

	@property
	def x(self):
		return self[0]

	@x.setter
	def x(self, value):
		self[0] = value

	@property
	def y(self):
		return self[1]

	@y.setter
	def y(self, value):
		self[1] = value

	@property
	def z(self):
		return self[2]

	@z.setter
	def z(self, value):
		self[2] = value

	def __eq__(self, other):
		return np.array_equal(self, other)

	def __ne__(self, other):
		return not np.array_equal(self, other)

	def abs_dot(self, other):
		return np.fabs(np.dot(self, other))

	def cross(self, other):
		return Vector(self.y * other.z - self.z * other.y,
		              self.z * other.x - self.x * other.z,
		              self.x * other.y - self.y * other.x)

	def sq_length(self):
		return self.x * self.x + self.y * self.y + self.z * self.z

	def length(self):
		return np.sqrt(self.sq_length())
