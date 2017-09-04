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

cdef class Shape:

	def __cinit__(self, Transform o2w, Transform w2o, bint ro):
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

	@staticmethod
	cdef inline void _incr_shape_id():
		global next_shape_id
		next_shape_id += 1

	@property
	def next_shape_id(self):
		return next_shape_id

	@property
	def shape_id(self):
		return self.shape_id

	cdef BBox _object_bound(self):
		raise NotImplementedError('unimplemented Shape.object_bound() method called')

	cdef BBox _world_bound(self):
		return self.o2w(self.object_bound())

	cdef bint _can_intersect(self):
		return True

	cdef void _refine(self, list refined):
		"""
		If `Shape` cannot intersect,
		return a refined subset
		"""
		raise NotImplementedError('Intersecable shapes are not refineable')

	cdef bint _intersect(self, Ray ray, FLOAT_t *thit, FLOAT_t *r_eps, DifferentialGeometry dg):
		raise NotImplementedError('unimplemented Shape.intersect() method called')

	cdef bint _intersect_p(self, Ray ray):
		raise NotImplementedError('unimplemented {}.intersect_p() method called'
		                          .format(self.__class__))

	cdef void _get_shading_geometry(self, Transform o2w, DifferentialGeometry dg, DifferentialGeometry dgs):
		pass

	cdef FLOAT_t _area(self):
		raise NotImplementedError('unimplemented Shape.area() method called')

	cdef void _sample(self, FLOAT_t u1, FLOAT_t u2, Point p, Normal n):
		"""
		_sample()

		Returns the position and
		surface normal randomly chosen
		"""
		raise NotImplementedError('unimplemented {}.sample() method called'
		                          .format(self.__class__))

	cdef void _sample_p(self, Point pnt, FLOAT_t u1, FLOAT_t u2, Point p, Normal n):
		"""
		_sample_p()

		Returns the position and
		surface normal randomly chosen
		s.t. visible to `pnt`
		"""
		self._sample(u1, u2, p, n)


	cdef FLOAT_t _pdf(self, Point pnt):
		"""
		_pdf()

		Return the sampling pdf
		"""
		return 1. / self._area()

	cdef FLOAT_t _pdf_p(self, Point pnt, Vector wi):
		"""
		_pdf_p()

		Return the sampling pdf w.r.t.
		the _sample_p() method.
		Transforms the density from one
		defined over area to one defined
		over solid angle from `pnt`.
		"""
		# intersect sample ray with area light
		cdef:
			Ray ray = Ray(pnt, wi, EPS)
			FLOAT_t thit = 0., r_eps = 0.
			DifferentialGeometry dg_light = DifferentialGeometry()

		if not self._intersect(ray, &thit, &r_eps, dg_light):
			return 0.

		# convert light sample weight to solid angle measure
		return (pnt - ray._at(thit))._sq_length() / (dg_light.nn._abs_dot(-wi) * self._area())
	

	cpdef BBox object_bound(self):
		return self._object_bound()

	cpdef BBox world_bound(self):
		return self._world_bound()

	cpdef bint can_intersect(self):
		return self._can_intersect()

	cpdef void refine(self, list refined):
		self._refine(refined)

	cpdef intersect(self, Ray ray):
		cdef:
			FLOAT_t thit = 0., r_eps = 0.
			DifferentialGeometry dg = DifferentialGeometry()
		if not self._intersect(ray, &thit, &r_eps, dg):
			return [False, 0., 0., None]
		return [True, 0., 0., None]

	cpdef bint intersect_p(self, Ray ray):
		return self._intersect_p(ray)

	cpdef FLOAT_t area(self):
		return self._area()

	cpdef list sample(self, FLOAT_t u1, FLOAT_t u2):
		cdef:
			Point p = Point()
			Normal n = Normal()

		self._sample(u1, u2, p, n)

		return [p, n]

	cpdef list sample_p(self, Point pnt, FLOAT_t u1, FLOAT_t u2):
		cdef:
			Point p = Point()
			Normal n = Normal()

		self._sample_p(pnt, u1, u2, p, n)

		return [p, n]

	cpdef FLOAT_t pdf(self, Point pnt):
		return self._pdf(pnt)

	cpdef FLOAT_t pdf_p(self, Point pnt, Vector wi):
		return self._pdf_p(pnt, wi)




