'''
test.py

A test script that (roghouly) test
the implementation of various classes

Created by Jiayao on 1 Aug, 2017
'''
import unittest as ut

from numba import jit   # jit support
import numpy as np      # numpy
from src.core.pytracer import * # globals

N_TEST_CASE = 100

from src.core.geometry import *
class test_geometry(ut.TestCase):
	def setUP(self):
		self.v = Vector(1., .5, 2.)

	def test_gloabl_methods(self):
		for _ in range(N_TEST_CASE):
			rd = np.random.rand(6)
			v = Vector(rd[0], rd[1], rd[2])
			n = Normal(rd[3], rd[4], rd[5])
			
			# coordinate_system()
			x, y, z = coordinate_system(v)
			self.assertAlmostEqual( FLOAT(x.dot(y)), 0., delta=EPS)
			self.assertAlmostEqual( FLOAT(y.dot(z)), 0., delta=EPS)
			self.assertAlmostEqual( FLOAT(z.dot(x)), 0., delta=EPS)
			
			# normalize()
			vn = normalize(v)
			self.assertAlmostEqual( FLOAT(vn.length()), 1., delta=EPS)

			# face_forward()
			vf = face_forward(n, v)
			self.assertGreaterEqual(n.dot(v), 0.)

			# spherical_direction
			theta = np.random.uniform(0., np.pi)
			phi = np.random.uniform(0., 2 * np.pi)

			vs = spherical_direction(np.sin(theta), np.cos(theta), phi)
			self.assertAlmostEqual(spherical_theta(vs), theta, delta=EPS)
			self.assertAlmostEqual(spherical_phi(vs), phi, delta=EPS)

	# test Vector, Point and Normal
	def test_class_vpn(self):
		rd = np.random.rand(9)
		v0 = Vector(rd[0], rd[1], rd[2])
		n0 = Normal(rd[3], rd[4], rd[5])
		p0 = Point(rd[6], rd[7], rd[8])

		# Vector, Normal and Point
		v1 = Vector.fromVector(v0)
		v2 = Vector.fromNormal(n0)
		v3 = Vector.fromPoint(p0)

		n1 = Normal.fromVector(v0)
		
		for i in range(3):
			self.assertEqual(v0[i], v1[i])
			self.assertEqual(v2[i], n0[i])
			self.assertEqual(v3[i], p0[i])
			self.assertEqual(n1[i], v0[i])

		fixture = [v0, p0, n0]
		for f in fixture:
			f.x, f.y, f.z = np.random.rand(3)
			self.assertEqual(f[0], f.x)
			self.assertEqual(f[1], f.y)
			self.assertEqual(f[2], f.z)

		p1 = Point()
		with self.assertRaises(TypeError):
			p0 -= p1

		o = Point()
		self.assertEqual(p0.sq_length(), p0.sq_dist(o))
		self.assertEqual(p0.length(), p0.dist(o))

		for _ in range(N_TEST_CASE):
			v0.x, v0.y, v0.z, v1.x, v1.y, v1.z = np.random.rand(6)
			p0.x, p0.y, p0.z, p1.x, p1.y, p1.z = np.random.rand(6)
			n0.x, n0.y, n0.z = np.random.rand(3)
			self.assertEqual(np.fabs(v0.dot(v1)), v0.abs_dot(v1))
			
			v2 = v0.cross(v1)
			self.assertAlmostEqual(v0.dot(v2), 0., delta=EPS)			
			self.assertAlmostEqual(v1.dot(v2), 0., delta=EPS)

			sq = v2.x ** 2 + v2.y ** 2 + v2.z ** 2
			self.assertAlmostEqual(v2.sq_length(), sq, delta=EPS)
			self.assertAlmostEqual(v2.length(), np.sqrt(sq), delta=EPS)

			sq = n0.x ** 2 + n0.y ** 2 + n0.z ** 2
			self.assertAlmostEqual(n0.sq_length(), sq, delta=EPS)
			self.assertAlmostEqual(n0.length(), np.sqrt(sq), delta=EPS)

			self.assertIsInstance(p0-p1, Vector)
			self.assertIsInstance(p0+p1, Point)
			self.assertIsInstance(p0+v1, Point)

			n1 = normalize(n0)
			n0.normalize()
			self.assertEqual(n0, n1)

	# Ray
	def test_class_ray(self):
		p = Point()
		v = Vector()
		ray = Ray(p, v)

		self.assertNotEqual(id(ray.o), id(p))
		self.assertNotEqual(id(ray.d), id(v))

		rc = Ray.fromParent(Point(1,2,3), Vector(4,5,6), ray)
		rr = Ray.fromRay(ray)
		rd = RayDifferential.fromRay(ray)

		self.assertEqual(ray.depth+1, rc.depth)
		self.assertEqual(ray.time, rc.time)
		self.assertEqual(ray.depth, rr.depth)
		self.assertNotEqual(id(ray.o), id(rr.o))
		self.assertNotEqual(id(ray.d), id(rr.d))
		self.assertNotEqual(id(ray.o), id(rc.o))
		self.assertNotEqual(id(ray.d), id(rc.d))			
		self.assertNotEqual(id(ray.o), id(rd.o))
		self.assertNotEqual(id(ray.d), id(rd.d))

		for _ in range(N_TEST_CASE):		
			p.x, p.y, p.z, v.x, v.y, v.z, mint, maxt, time = np.random.rand(9)
			dp = np.random.randint(N_TEST_CASE)

			ray = Ray(p, v, mint, maxt, dp, time)
			pr = ray(time)
			self.assertEqual(pr, ray.o + ray.d * time)

	# BBox
	def test_bbox(self):
		o = Point()
		p0 = Point(1,1,1)
		p1 = Point(2,2,2)
		b0 = BBox(p1)
		b1 = BBox(None, p0)
		b2 = BBox(o, p1)
		b3 = BBox()
		b4 = BBox.fromBBox(b2)

		self.assertNotEqual(id(b2.pMin), id(b4.pMin))
		self.assertNotEqual(id(b2.pMax), id(b4.pMax))
		self.assertEqual(id(b1[0]), id(b1.pMin))
		self.assertEqual(id(b1[1]), id(b1.pMax))

		b0 = BBox(o, p0)
		b1 = BBox(p1, p0)
		self.assertEqual(b0.union(b1), b4)
		self.assertEqual(b0.union(p1), b4)
		self.assertEqual(b0.union(b1), BBox.Union(b0, b1))
		self.assertNotEqual(id(b0.union(b1)), id(BBox.Union(b0, b1)))

		b5 = b2
		b6 = BBox(-p1, -p0)
		self.assertEqual(b5.overlaps(b4), True)
		self.assertEqual(b6.overlaps(b5), False)
		self.assertEqual(b5.inside(p0), True)
		self.assertEqual(b5.surface_area(), 24.)
		self.assertEqual(b5.volume(), 8.)		
		
		for i in range(3):
			p = Point(1., 1., 1.,)
			p[i] = 2.
			b7 = BBox(o, p)
			self.assertEqual(b7.maximum_extent(), i)

		b8 = BBox(o, p0)
		for _ in range(N_TEST_CASE):
			p0.x, p0.y, p0.z = np.random.uniform(0., 2., 3)
			tx, ty, tz = np.random.uniform(0., 1., 3)
			p = b8.lerp(tx, ty, tz)
			self.assertAlmostEqual(p.sq_length(), tx ** 2 + ty ** 2 + tz ** 2, delta=EPS)
			v = b8.offset(p)
			self.assertAlmostEqual(v[0], tx, delta=EPS)
			self.assertAlmostEqual(v[1], ty, delta=EPS)
			self.assertAlmostEqual(v[2], tz, delta=EPS)

		ctr, rad = b8.bounding_sphere()
		self.assertEqual(ctr, Point(.5, .5, .5))
		self.assertAlmostEqual(rad, np.sqrt(3) / 2, delta=EPS)


		b9 = BBox(Point(1.,1.,1.), Point(2.,2.,2.))
		
		r1 = Ray(o, Vector(1., 1., 1.))

		isec, t1, t2 = b9.intersectP(r1)
		self.assertEqual(isec, True)		
		self.assertAlmostEqual(t1, 1., delta=EPS)
		self.assertAlmostEqual(t2, 2., delta=EPS)

		# r2 = Ray(Point(1., 1., 0.), Vector(0., 0., 1.))
		# with self.assertRaises(RuntimeWarning):
		# 	isec, t1, t2 = b9.intersectP(r2)
		# self.assertEqual(isec, True)				
		# self.assertAlmostEqual(t1, 1., delta=EPS)
		# self.assertAlmostEqual(t2, 2., delta=EPS)	

		r3 = Ray(Point(1., 1., 1.), Vector(-1., -1., -1.))
		isec, t1, t2 = b9.intersectP(r3)
		self.assertEqual(isec, True)		
		self.assertAlmostEqual(t1, 0., delta=EPS)
		self.assertAlmostEqual(t2, 0., delta=EPS)	

		r4 = Ray(Point(), Vector(-1., -1., -1.))
		isec, t1, t2 = b9.intersectP(r4)
		self.assertEqual(isec, False)		
		self.assertAlmostEqual(t1, 0., delta=EPS)
		self.assertAlmostEqual(t2, 0., delta=EPS)	


# def test():
# 	p0 = Point(1,1,1)
# 	p1 = Point(2,3,5)
# 	v = Vector(1,2,3)
# 	a = Transform.look_at(p0, p1, v)
# 	b = Transform.rotate(36, v)
# 	assert feq(np.linalg.inv(a.mInv),a.m).all()
# 	assert feq(np.linalg.inv(b.mInv),b.m).all()


from src.core.spectrum import *
def test_spect():
	assert feq(SampledSpectrum.average_spectrum_samples([1,2,3,4],[3,4,5,3],4,1.5,3.5),4.3125)
	assert feq(SampledSpectrum.average_spectrum_samples([1,2,3,4],[3,4,5,3],4,0.5,4), 3.8571428571428571)
	assert feq(SampledSpectrum.average_spectrum_samples([1,2,3,4],[3,4,5,3],4,0.5,4.5), 3.75)