"""
__init__.py

pytracer.geometry package

Created by Jiayao on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer import *

# Classes


class Vector(np.ndarray):
	"""
	Vector Class

	A wrappper subclasses numpy.ndarray which
	models a 3D vector.
	"""

	def __new__(cls, x: FLOAT = 0., y: FLOAT = 0., z: FLOAT = 0, dtype=FLOAT):
		return np.empty(3, dtype=dtype).view(cls)

	def __init__(self, x: FLOAT = 0., y: FLOAT = 0., z: FLOAT = 0., dtype=FLOAT):
		if np.isnan(x) or np.isnan(y) or np.isnan(z):
			raise ValueError
		super().__init__()
		self.x = x
		self.y = y
		self.z = z

	@classmethod
	def from_arr(cls, n: 'np.ndarray'):  # Forward type hint (PEP-484)
		assert np.shape(n) == 3
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

	def abs_dot(self, other) -> FLOAT:
		return np.fabs(np.dot(self, other))

	def cross(self, other) -> 'Vector':
		return Vector(self.y * other.z - self.z * other.y,
		              self.z * other.x - self.x * other.z,
		              self.x * other.y - self.y * other.x)

	def sq_length(self) -> FLOAT:
		return self.x * self.x + self.y * self.y + self.z * self.z

	def length(self) -> FLOAT:
		return np.sqrt(self.sq_length())


class Point(np.ndarray):
	"""
	Point class

	A wrappper subclasses numpy.ndarray which
	models a 3D vector. Note the subtle difference
	between vector.

	The defaul inheritance behavior is consistent
	with the notion of Point and Vector arithmetic.
	The returining type is the same as the type
	of the first operant, thus we may, e.g., offset
	a point by a vector.
	"""
	def __new__(cls, x: FLOAT = 0., y: FLOAT = 0., z: FLOAT = 0, dtype=FLOAT):
		return np.empty(3, dtype=dtype).view(cls)

	def __init__(self, x: FLOAT = 0., y: FLOAT = 0., z: FLOAT = 0., dtype=FLOAT):
		if np.isnan(x) or np.isnan(y) or np.isnan(z):
			raise ValueError
		super().__init__()
		self.x = x
		self.y = y
		self.z = z

	@classmethod
	def from_arr(cls, n: 'np.ndarray'):  # Forward type hint (PEP-484)
		assert np.shape(n) == 3
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

	def __sub__(self, other):
		if isinstance(other, Point):  # no other methods found
			return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

		elif isinstance(other, Vector):
			return Point(self.x - other.x, self.y - other.y, self.z - other.z)

		else:
			raise TypeError("unsupported __sub__ between '{}' and '{}'".format(self.__class__, type(other)))

	def __isub__(self, other):
		raise TypeError("undefined inplace substraction between Point")

	# addition, however, is defined, as can be used for weighing points

	def sq_length(self) -> FLOAT:
		return self.x * self.x + self.y * self.y + self.z * self.z

	def length(self) -> FLOAT:
		return np.sqrt(self.sq_length())

	def sq_dist(self, other) -> FLOAT:
		return (self.x - other.x) * (self.x - other.x) + \
		       (self.y - other.y) * (self.y - other.y) + \
		       (self.z - other.z) * (self.z - other.z)

	def dist(self, other) -> FLOAT:
		return np.sqrt((self.x - other.x) * (self.x - other.x) +
		               (self.y - other.y) * (self.y - other.y) +
		               (self.z - other.z) * (self.z - other.z))


class Normal(np.ndarray):
	"""
	Normal vector class

	A wrapper subclasses numpy.ndarray which
	models a 3D vector.
	"""
	def __new__(cls, x: FLOAT = 0., y: FLOAT = 0., z: FLOAT = 0, dtype=FLOAT):
		return np.empty(3, dtype=dtype).view(cls)

	def __init__(self, x: FLOAT = 0., y: FLOAT = 0., z: FLOAT = 0., dtype=FLOAT):
		super().__init__()
		if np.isnan(x) or np.isnan(y) or np.isnan(z):
			raise ValueError
		self.x = x
		self.y = y
		self.z = z

	@classmethod
	def from_arr(cls, n: 'np.ndarray'):  # Forward type hint (PEP-484)
		assert np.shape(n) == 3
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

	def cross(self, other) -> Vector:
		return Vector(self.y * other.z - self.z * other.y,
		              self.z * other.x - self.x * other.z,
		              self.x * other.y - self.y * other.x)

	def sq_length(self) -> FLOAT:
		return self.x * self.x + self.y * self.y + self.z * self.z

	def length(self) -> FLOAT:
		return np.sqrt(self.sq_length())

	def normalize(self):
		"""inplace normalization"""
		length = self.length()
		if not length == 0:
			self /= length
		return self


class Ray(object):
	"""
	Ray class

	Models a semi-infinite ray as parametric line with
	starting `Point`, direction `Vector`, visual range from
	`mint` to `maxt`, the bouncing `depth` and animated `time`.
	"""

	def __init__(self, o: 'Point'=Point(0., 0., 0.), d: 'Vector'=Vector(0., 0., 0.),
	             mint: FLOAT = 0., maxt: FLOAT = np.inf,
	             depth: INT = 0, time: FLOAT = 0.):
		self.o = Point(o.x, o.y, o.z)
		self.d = Vector(d.x, d.y, d.z)
		self.mint = mint
		self.maxt = maxt
		self.depth = depth
		self.time = time  # for motion blur or animation purposes

	def __repr__(self):
		return "{}\nOrigin:    {}\nDirection: {}\nmint: {}    \
				maxt: {}\ndepth: {}\ntime: {}".format(self.__class__,
		        self.o, self.d, self.mint, self.maxt, self.depth, self.time)

	@classmethod
	def from_parent(cls, o: 'Point', d: 'Vector', r: 'Ray',
	               mint: FLOAT = 0., maxt: FLOAT = np.inf):
		"""
		initialize from a parent ray
		"""
		return cls(o, d, mint, maxt, r.depth + 1, r.time)

	@classmethod
	def from_ray(cls, r: 'Ray'):
		"""
		initialize from a ray, analogous to a copy constructor
		"""
		return cls(r.o, r.d, r.mint, r.maxt, r.depth, r.time)

	def __call__(self, t) -> 'Point':
		"""
		point at parameter t
		"""
		return (self.o + self.d * t).view(Point)


class RayDifferential(Ray):
	"""
	RayDifferential Class

	Subclasses `Ray` for texture mapping
	"""

	def __init__(self, o: 'Point'=Point(0, 0, 0), d: 'Vector'=Vector(0, 0, 0),
	             mint: FLOAT = 0., maxt: FLOAT = np.inf,
	             depth: INT = 0, time: FLOAT = 0.):
		super().__init__(o, d, mint, maxt, depth, time)
		self.has_differentials = False
		self.rxOrigin = Point(0., 0., 0.)
		self.ryOrigin = Point(0., 0., 0.)
		self.rxDirection = Vector(0., 0., 0.)
		self.ryDirection = Vector(0., 0., 0.)

	@classmethod
	def from_parent(cls, o: 'Point', d: 'Vector', r: 'RayDifferential',
	               mint: FLOAT=0., maxt: FLOAT =np.inf):
		"""
		initialize from a parent ray
		"""
		return cls(o, d, mint, maxt, r.depth + 1, r.time)

	@classmethod
	def from_ray(cls, r: 'Ray'):
		"""
		initialize from a ray, analogous to a copy constructor
		"""
		return cls(r.o, r.d, r.mint, r.maxt, r.depth, r.time)

	@classmethod
	def from_rd(cls, r: 'RayDifferential'):
		"""
		initialize from a `RayDifferential`, analogous to a copy constructor
		"""
		self = cls(r.o, r.d, r.mint, r.maxt, r.depth, r.time)
		self.has_differentials = r.has_differentials
		self.rxOrigin = r.rxOrigin.copy()
		self.ryOrigin = r.ryOrigin.copy()
		self.rxDirection = r.rxDirection.copy()
		self.ryDirection = r.ryDirection.copy()

		return self

	def scale_differential(self, s: FLOAT):
		self.rxOrigin = self.o + (self.rxOrigin - self.o) * s
		self.ryOrigin = self.o + (self.ryOrigin - self.o) * s
		self.rxDirection = self.d + (self.rxDirection - self.d) * s
		self.ryDirection = self.d + (self.ryDirection - self.d) * s
		return self


class BBox:
	"""
	BBox Class

	Models 3D axis-aligned bounding boxes
	represented by a pair of opposite vertices.
	"""

	def __init__(self, p1=None, p2=None):
		if p1 is not None and p2 is not None:
			self.pMin = Point(min(p1.x, p2.x),
			                  min(p1.y, p2.y),
			                  min(p1.z, p2.z))
			self.pMax = Point(max(p1.x, p2.x),
			                  max(p1.y, p2.y),
			                  max(p1.z, p2.z))

		# default: degenerated BBox
		elif p1 is None and p2 is None:
			self.pMin = Point(np.inf, np.inf, np.inf)
			self.pMax = Point(-np.inf, -np.inf, -np.inf)

		elif p2 is None:
			self.pMin = p1.copy()
			self.pMax = Point(np.inf, np.inf, np.inf)

		else:
			self.pMin = Point(-np.inf, -np.inf, -np.inf)
			self.pMax = p2.copy()

	@classmethod
	def from_bbox(cls, box: 'BBox'):
		return cls(box.pMin, box.pMax)

	def __repr__(self):
		return "{}\npMin:{}\npMax:{}".format(self.__class__,
		                                     self.pMin,
		                                     self.pMax)

	def __eq__(self, other):
		return np.array_equal(self.pMin, other.pMin) and \
		       np.array_equal(self.pMax, other.pMax)

	def __ne__(self, other):
		return not np.array_equal(self.pMin, other.pMax) or \
		       not np.array_equal(self.pMin, other.pMax)

	def __getitem__(self, key):
		if key == 0:
			return self.pMin
		elif key == 1:
			return self.pMax
		else:
			raise KeyError

	@staticmethod
	def Union(b1, b2) -> 'BBox':
		"""
		Return the union of a `BBox`
		and a `Point` or a union
		of two `Box`es.
		"""
		ret = BBox()

		if isinstance(b2, Point):
			ret.pMin.x = min(b1.pMin.x, b2.x)
			ret.pMin.y = min(b1.pMin.y, b2.y)
			ret.pMin.z = min(b1.pMin.z, b2.z)
			ret.pMax.x = max(b1.pMax.x, b2.x)
			ret.pMax.y = max(b1.pMax.y, b2.y)
			ret.pMax.z = max(b1.pMax.z, b2.z)

		elif isinstance(b2, BBox):
			ret.pMin.x = min(b1.pMin.x, b2.pMin.x)
			ret.pMin.y = min(b1.pMin.y, b2.pMin.y)
			ret.pMin.z = min(b1.pMin.z, b2.pMin.z)
			ret.pMax.x = max(b1.pMax.x, b2.pMax.x)
			ret.pMax.y = max(b1.pMax.y, b2.pMax.y)
			ret.pMax.z = max(b1.pMax.z, b2.pMax.z)

		else:
			raise TypeError('unsupported union operation between\
							{} and {}'.format(type(b1), type(b2)))
		return ret

	def union(self, other) -> 'BBox':
		"""
		Return self as the union of a `BBox`
		and a `Point` or a union
		of two `Box`es.
		"""

		if isinstance(other, Point):
			self.pMin.x = min(self.pMin.x, other.x)
			self.pMin.y = min(self.pMin.y, other.y)
			self.pMin.z = min(self.pMin.z, other.z)
			self.pMax.x = max(self.pMax.x, other.x)
			self.pMax.y = max(self.pMax.y, other.y)
			self.pMax.z = max(self.pMax.z, other.z)

		elif isinstance(other, BBox):
			self.pMin.x = min(self.pMin.x, other.pMin.x)
			self.pMin.y = min(self.pMin.y, other.pMin.y)
			self.pMin.z = min(self.pMin.z, other.pMin.z)
			self.pMax.x = max(self.pMax.x, other.pMax.x)
			self.pMax.y = max(self.pMax.y, other.pMax.y)
			self.pMax.z = max(self.pMax.z, other.pMax.z)

		else:
			raise TypeError('unsupported union operation between\
							{} and {}'.format(self.__class__, type(other)))
		return self

	def overlaps(self, other: 'BBox') -> bool:
		"""
		Determines whether two `BBox`es overlaps
		"""
		return (self.pMax.x >= other.pMin.x) and (self.pMin.x <= other.pMax.x) and \
		       (self.pMax.y >= other.pMin.y) and (self.pMin.y <= other.pMax.y) and \
		       (self.pMax.z >= other.pMin.z) and (self.pMin.z <= other.pMax.z)

	def inside(self, pnt: 'Point') -> bool:
		"""
		Determines whether a given `Point`
		is inside the box
		"""
		return (self.pMax.x >= pnt.x) and (self.pMin.x <= pnt.x) and \
		       (self.pMax.y >= pnt.y) and (self.pMin.y <= pnt.y) and \
		       (self.pMax.z >= pnt.z) and (self.pMin.z <= pnt.z)

	def expand(self, delta: FLOAT) -> 'BBox':
		"""
		Expands box by a constant factor
		"""
		self.pMin.x -= delta
		self.pMin.y -= delta
		self.pMin.z -= delta
		self.pMax.x += delta
		self.pMax.y += delta
		self.pMax.z += delta
		return self

	def surface_area(self) -> FLOAT:
		"""
		Computes the surface area
		"""
		d = (self.pMax - self.pMin).view(Vector)
		return 2. * (d.x * d.y + d.x * d.z + d.y * d.z)

	def volume(self) -> FLOAT:
		"""
		Computes the volume
		"""
		d = (self.pMax - self.pMin).view(Vector)
		return d.x * d.y * d.z

	def maximum_extent(self):
		"""
		Find the maximum axis
		"""
		delta = (self.pMax - self.pMin).view(Vector)
		if delta.x > delta.y and delta.x > delta.z:
			return 0
		elif delta.y > delta.z:
			return 1
		else:
			return 2

	def lerp(self, tx: FLOAT, ty: FLOAT, tz: FLOAT) -> 'Point':
		"""
		lerp
		3D Linear interpolation between two opposite vertices
		"""
		return Point(util.lerp(tx, self.pMin.x, self.pMax.x),
		             util.lerp(ty, self.pMin.y, self.pMax.y),
		             util.lerp(tz, self.pMin.z, self.pMax.z))

	def offset(self, pnt: 'Point') -> 'Vector':
		"""
		offset
		Get point relative to the corners
		"""
		return Vector((pnt.x - self.pMin.x) / (self.pMax.x - self.pMin.x),
		              (pnt.y - self.pMin.y) / (self.pMax.y - self.pMin.y),
		              (pnt.z - self.pMin.z) / (self.pMax.z - self.pMin.z))

	def bounding_sphere(self) -> ('Point', FLOAT):
		"""
		bounding_sphere
		Get the center and radius of the bounding sphere
		"""
		ctr = (.5 * (self.pMin + self.pMax)).view(Point)
		rad = 0.
		if self.inside(ctr):
			rad = ctr.dist(self.pMax)
		return ctr, rad

	def intersect_p(self, r: 'Ray') -> [bool, FLOAT, FLOAT]:
		"""
		intersect_p()

		Check whether a ray intersects the BBox
		Compare intervals along each dimension
		returns the shortest/largest parametric values
		if intersects.
		"""
		t0, t1 = r.mint, r.maxt
		# automatically convert /0. to np.inf
		t_near = np.true_divide((self.pMin - r.o), r.d)
		t_far = np.true_divide((self.pMax - r.o), r.d)
		for i in range(3):
			if t_near[i] > t_far[i]:
				t_near[i], t_far[i] = t_far[i], t_near[i]
			t0, t1 = np.fmax(t0, t_near[i]), np.fmin(t1, t_far[i])
			if t0 > t1:
				return [False, 0., 0.]
		return [True, t0, t1]


from pytracer.geometry.diffgeom import DifferentialGeometry
from pytracer.geometry.utility import *