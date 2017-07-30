'''
geometry.py

This module is part of the pyTracer, which
defines the geometric classes.

v0.0
Created by Jiayao on July 27, 2017
'''
import numpy as np
from src.core.pytracer import *

# Geometry related functions
def coordinate_system(v1: 'Vector') -> ['Vector']:
	'''
	construct a left-handed coordinate system
	with v1, which is assumed to be normalized.
	'''
	if(np.fabs(v1.x) > np.fabs(v1.y)):
		invLen = 1. / np.sqrt(v1.x * v1.x + v1.z * v1.z)
		v2 = Vector(-v1.z * invLen, 0., v1.x * invLen)

	else:
		invLen = 1. / np.sqrt(v1.y * v1.y + v1.z * v1.z)
		v2 = Vector(0., -v1.z * invLen, v1.y * invLen)

	v3 = v1.cross(v2)

	return v1, v2, v3

def normalize(vec: 'Vector') -> 'Vector':
	'''
	returns a new normalized vector
	'''
	n = vec.copy()
	length = vec.length()
	if not length == 0:
		n /= length
	return n	

def face_forward(n: 'Normal', v: 'Vector') -> 'Vector':
	'''
	Flip a normal according to a vector
	'''
	if not isinstance(n, Normal) or not isinstance(v, Vector):
		raise TypeError('Argument type error')

	if (np.dot(n, v) < 0.):
		return -n.copy()

	return n.copy()

# Classes
class Vector(np.ndarray):
	'''
	Vector class

	A wrappper subclasses numpy.ndarray which 
	models a 3D vector.
	'''
	def __new__(cls, x:FLOAT=0., y:FLOAT=0., z:FLOAT=0., dtype=FLOAT):
		if np.isnan(x) or np.isnan(y) or np.isnan(z):
			raise ValueError
		obj = np.asarray([x, y, z], dtype=dtype).view(cls)
		return obj

	@classmethod
	def fromNormal(cls, n: 'Normal'): # Forward type hint, see PEP-484
		return cls(n.x, n.y, n.z)

	@classmethod
	def fromPoint(cls, n: 'Point'):
		return cls(n.x, n.y, n.z)				

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
	'''
	Point class

	A wrappper subclasses numpy.ndarray which 
	models a 3D vector. Note the subtle difference
	between vector.

	The defaul inheritance behavior is consistent
	with the notion of Point and Vector arithmetic.
	The returining type is the same as the type
	of the first operant, thus we may, e.g., offset
	a point by a vector.
	'''
	def __new__(cls, x:FLOAT=0., y:FLOAT=0., z:FLOAT=0., dtype=FLOAT):
		if np.isnan(x) or np.isnan(y) or np.isnan(z):
			raise ValueError
		obj = np.asarray([x, y, z], dtype=dtype).view(cls)
		return obj		

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
		if isinstance(other, Point): # no other methods found
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
		return (self - other).sq_length()
	
	def dist(self, other) -> FLOAT:
		return (self - other).length()

class Normal(np.ndarray):
	'''
	Normal vector class

	A wrappper subclasses numpy.ndarray which 
	models a 3D vector.
	'''
	def __new__(cls, x:FLOAT=0., y:FLOAT=0., z:FLOAT=0., dtype=FLOAT):
		if np.isnan(x) or np.isnan(y) or np.isnan(z):
			raise ValueError
		obj = np.asarray([x, y, z], dtype=dtype).view(cls)
		return obj
	
	@classmethod
	def fromVector(cls, vec: 'Vector'):
		return cls(vec.x, vec.y, vec.z)
	
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
		'''
		inplace normalization
		'''
		length = self.length()
		if not length == 0:
			self /= length

class Ray(object):
	'''
	Ray class

	Models a semi-infimite ray as parametric line with
	starting `Point`, direction `Vector`, visual range from
	`mint` to `maxt`, the bouncing `depth` and animated `time`.
	'''
	def __init__(self, o: Point=Point(0,0,0), d: Vector=Vector(0,0,0), 
			mint: FLOAT=0., maxt: FLOAT=np.inf, 
			depth: INT=0, time: FLOAT=0.):
		self.o = o.copy() # NB: using copy
		self.d = d.copy()
		self.mint = mint
		self.maxt = maxt
		self.depth = depth
		self.time = time # for motion blur or animation purposes
	
	def __repr__(self):
		return "{}\nOrigin:    {}\nDirection: {}\nmint: {}    \
				maxt: {}\ndepth: {}\ntime: {}".format(self.__class__,
					self.o, self.d, self.mint, self.maxt, self. depth,
					self.time)
	
	@classmethod
	def fromParent(cls, o: 'Point', d: 'Vector', r:'Ray',
			mint: FLOAT, maxt: FLOAT=np.inf):
		'''
		initialize from a parent ray
		'''
		return cls(o, d, mint, maxt, r.depth + 1, r.time)

	@classmethod
	def fromRay(cls, r: 'Ray'):
		'''
		initialize from a ray, analogous to a copy constructor
		'''
		return cls(r.o, r.d, r.mint, r.maxt, r.depth, r.time)
	
	def __call__(self, t) -> 'Point':
		'''
		point at parameter t
		'''
		return self.o + self.d * t

class RayDifferential(Ray):
	'''
	RayDifferential Class

	Subclasses `Ray` for texture mapping
	'''
	def __init__(self, o: Point=Point(0,0,0), d: Vector=Vector(0,0,0), 
			mint: FLOAT=0., maxt: FLOAT=np.inf, 
			depth: INT=0, time: FLOAT=0.):
		super().__init__(o, d, mint, maxt, depth, time)
		self.hasDifferentials = False
		self.rxOrigin = Point()
		self.ryOrigin = Point()
		self.rxDirection = Vector()
		self.ryDirection = Vector()
	
	@classmethod
	def fromParent(cls, o: Point, d: Vector, r:Ray,
			mint: FLOAT, maxt: FLOAT=np.inf):
		'''
		initialize from a parent ray
		'''
		return cls(o, d, mint, maxt, r.depth + 1, r.time)
	
	@classmethod
	def fromRay(cls, r: Ray):
		'''
		initialize from a ray, analogous to a copy constructor
		'''
		return cls(r.o, r.d, r.mint, r.maxt, r.depth, r.time)
	
	def scale_differential(self, s:FLOAT):
		self.rxOrigin = o + (rxOrigin - o) * s
		self.ryOrigin = o + (ryOrigin - o) * s
		self.rxDirection = d + (rxDirection - d) * s
		self.ryDirection = d + (ryDirection - d) * s

class BBox:
	'''
	BBox Class

	Models 3D axis-aligned bounding boxes
	represented by a pair of opposite vertices.
	'''
	def __init__(self, p1=None, p2=None):
		if not p1 is None and not p2 is None:
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
		
		else:
			if not p2:
				p1 = p2
			self.pMin = p1.copy()
			self.pMax = p1.copy()
	
	@classmethod
	def fromBBox(cls, box: 'BBox'):
		return cls(box.pMin, box.pMax)
	
	def __repr__(self):
		return "{}\npMin:{}\npMax:{}".format(self.__class__,
											 self.pMin,
											 self.pMax)
	def __eq__(self, other):
		return np.array_equal(self.pMin, other.pMax) and \
			   np.array_equal(self.pMin, other.pMax)

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
		'''
		Return the union of a `BBox`
		and a `Point` or a union
		of two `Box`es.
		'''	
		ret = BBox()

		if isinstance(other, Point):
			ret.pMin.x = min(b1.pMin.x, b2.x)
			ret.pMin.y = min(b1.pMin.y, b2.y)
			ret.pMin.z = min(b1.pMin.z, b2.z)
			ret.pMax.x = max(b1.pMax.x, b2.x)
			ret.pMax.y = max(b1.pMax.y, b2.y)
			ret.pMax.z = max(b1.pMax.z, b2.z)
		
		elif isinstance(other, BBox):
			ret.pMin.x = min(b1.pMin.x, b2.pMin.x)
			ret.pMin.y = min(b1.pMin.y, b2.pMin.y)
			ret.pMin.z = min(b1.pMin.z, b2.pMin.z)
			ret.pMax.x = max(b1.pMax.x, b2.pMax.x)
			ret.pMax.y = max(b1.pMax.y, b2.pMax.y)
			ret.pMax.z = max(b1.pMax.z, b2.pMax.z)
		
		else:
			raise TypeError('unsupported union operation between\
							{} and {}'.format(self.__class__, type(other)))			
		return ret		
	
	def union(self, other) -> 'BBox':
		'''
		Return self as the union of a `BBox`
		and a `Point` or a union
		of two `Box`es.
		'''
		
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
		'''
		Determines whether two `BBox`es overlaps
		'''
		return (self.pMax.x >= other.pMin.x) and (self.pMin.x <= other.pMax.x) and \
			   (self.pMax.y >= other.pMin.y) and (self.pMin.y <= other.pMax.y) and \
			   (self.pMax.z >= other.pMin.z) and (self.pMin.z <= other.pMax.z)
	
	def inside(self, pnt: 'Point') -> bool:
		'''
		Determines whether a given `Point`
		is inside the box
		'''
		return (self.pMax.x >= pnt.x) and (self.pMin.x <= pnt.x) and \
			   (self.pMax.y >= pnt.y) and (self.pMin.y <= pnt.y) and \
			   (self.pMax.z >= pnt.z) and (self.pMin.z <= pnt.z)
	
	def expand(self, delta: FLOAT) -> None:
		'''
		Expands box by a constant factor
		'''
		self.pMin.x -= delta
		self.pMin.y -= delta
		self.pMin.z -= delta
		self.pMax.x += delta
		self.pMax.y += delta
		self.pMax.z += delta
	
	def surface_area(self) -> FLOAT:
		'''
		Computes the surface area
		'''
		delta = self.pMax - self.pMin
		return 2. * (d.x * d.y + d.x * d.z + d.y * d.z)
	
	def volume(self) -> FLOAT:
		'''
		Computes the volume
		'''
		delta = self.pMax - self.pMin
		return d.x * d.y * d.z
	
	def maximum_extent(self):
		'''
		Find the maximum axis
		'''
		delta = self.pMax - self.pMin
		if delta.x > delta.y and delta.x > delta.z:
			return 0
		elif delta.y > delta.z:
			return 1
		else:
			return 2
	
	def lerp(self, tx: FLOAT, ty: FLOAT, tz: FLOAT) -> 'Point':
		'''
		Lerp
		3D Linear interpolation between two opposite vertices
		'''
		return Point(Lerp(tx, self.pMin.x, self.pMax.x),
					 Lerp(ty, self.pMin.y, self.pMax.y),
					 Lerp(tz, self.pMin.z, self.pMax.z))
	
	def offset(self, pnt: 'Point') -> 'Vector':
		'''
		offset
		Get point relative to the corners
		'''
		return Vector((pnt.x - self.pMin.x) / (self.pMax.x - self.pMin.x),
					  (pnt.y - self.pMin.y) / (self.pMax.y - self.pMin.y),
					  (pnt.z - self.pMin.z) / (self.pMax.z - self.pMin.z))
	
	def bounding_sphere(self) -> ('Point', FLOAT):
		'''
		bounding_sphere
		Get the center and radius of the bounding sphere
		'''
		ctr = .5 * (self.pMin + self.pMax)
		rad = 0.
		if self.inside(ctr):
			rad = ctr.dist(self.pMax)
		return ctr, rad

	def intersectP(self, r: 'Ray') -> [bool, FLOAT, FLOAT]:
		t0, t1 = r.mint, r.maxt
		dInv = 1. / r.d
		tNear = (pMin - ray.o) * dInv
		tFar = (pMax - ray.o) * dInv
		for i in range(3):
		if tNear[i] > tFar[i]:
			tNear[i], tFar[i] = tFar[i], tNear[i]
			t0, t1 = max(t0, tNear), max(t1, tFar)
			if t0 > t1:
				return False
		return [True, t0, t1]




