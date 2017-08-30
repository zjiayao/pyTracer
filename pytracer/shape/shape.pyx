"""
shape.pxd

pytracer.shape package

Contains the base interface for shapes.

Implementation includes:
	- LoopSubdiv
	- TriangleMesh
	- Sphere
	- Cylinder
	- Disk

Created by Jiayao on July 27, 2017
Modified on Aug 13, 2017
Cythonized on Aug 30, 2017
"""

cdef INT_t next_shape_id = 1

cdef class Shape(metaclass=ABCMeta):

	def __cinit__(self, Transform o2w, Transform w2o, bool ro):
		self.o2w = o2w
		self.w2o = w2o
		self.ro = ro
		self.transform_swaps_handedness = o2w.swaps_handedness()

		self.shape_id = next_shape_id
		Shape._incr_shape_id()

	def __repr__(self):
		global next_shape_id
		return "{}\nInstance Count: {}\nShape Id: {}" \
			.format(self.__class__, next_shape_id, self.shape_id)

	@property
	def next_shape_id(self):
		return next_shape_id

	@property
	def shape_id(self):
		return self.shape_id

	cpdef BBox object_bound(self):
		return self._object_bound()

	cpdef BBox world_bound(self):
		return self._world_bound()

	cpdef bint can_intersect(self):
		return self._can_intersect()

	cpdef void refine(self, cppvector[Shape]* refined):
		return self._refine(refined)

	cpdef intersect(self, Ray ray):
		cdef:
			FLOAT_t thit, r_eps
			DifferentialGeometry dg
		if not self._intersect(ray, &thit, &r_eps, &dg):
			return [False, 0., 0., None]
		return [True, 0., 0., None]

	cpdef bint intersect_p(self, Ray ray):
		return self._intersect_p(ray)

	cpdef FLOAT_t area(self):
		return self._area()

	cpdef sample(self, FLOAT_t u1, FLOAT_t u2):
		cdef:
			Point p
			Normal n

		self._sample(u1, u2, p, n)

		return [p, n]

	cpdef sample_p(self, Point pnt, FLOAT_t u1, FLOAT_t u2):
		cdef:
			Point p
			Normal n

		self._sample_p(pnt, u1, u2, p, n)

		return [p, n]

	cpdef FLOAT_t pdf(self, Point pnt):
		return self._pdf(pnt)

	cpdef FLOAT_t pdf_p(self, Point pnt, Vector wi):
		return self._pdf_p(pnt, wi)




