"""
test_geometry.py

A test script that (roughly) test
the implementation of various classes

Created by Jiayao on 1 Aug, 2017
"""
from __future__ import absolute_import

import numpy as np
import pytest
from pytracer import EPS
import pytracer.geometry as geo

N_TEST_CASE = 5
np.random.seed(1)
rng = np.random.rand

testdata = {
	'vector': [geo.Vector(rng(), rng(), rng()) for _ in range(N_TEST_CASE)],
	'normal': [geo.Normal(rng(), rng(), rng()) for _ in range(N_TEST_CASE)],
	'point': [geo.Point(rng(), rng(), rng()) for _ in range(N_TEST_CASE)],
	'theta': [np.random.uniform(0., np.pi) for _ in range(N_TEST_CASE)],
	'phi': [np.random.uniform(0., 2 * np.pi) for _ in range(N_TEST_CASE)],

}


def assert_almost_eq(a, b, thres=EPS):
	assert a == pytest.approx(b, abs=thres)


def assert_triple_eq(a: 'np.ndarray', b: 'np.ndarray', thres=EPS):
	assert a == pytest.approx(b, abs=thres)


def assert_elem_eq(a: 'np.ndarray', b: 'np.ndarray', thres=EPS):
	assert np.shape(a) == np.shape(b)
	for i in range(len(a)):
		assert_almost_eq(a[i], b[i])


class TestGeometry(object):

	@pytest.mark.parametrize("vec", testdata['vector'])
	def test_coordinate_system(self, vec):
		x, y, z = geo.coordinate_system(vec)
		assert_almost_eq(x.dot(y), 0.)
		assert_almost_eq(x.dot(z), 0.)
		assert_almost_eq(y.dot(z), 0.)

	@pytest.mark.parametrize("vec", testdata['vector'])
	def test_normalize(self, vec):
		v = geo.normalize(vec)
		assert_triple_eq(v / v.length(), vec)

	@pytest.mark.parametrize("vec", testdata['vector'])
	@pytest.mark.parametrize("norm", testdata['normal'])
	def test_face_forward(self, vec, norm):
		n = geo.face_forward(norm, vec)
		assert n.dot(vec) >= 0.

	@pytest.mark.parametrize("theta", testdata['phi'])
	@pytest.mark.parametrize("phi", testdata['phi'])
	def test_spherical(self, theta, phi):
		st = np.sin(theta)
		ct = np.cos(theta)
		v = geo.spherical_direction(st, ct, phi)
		theta_ret = geo.spherical_theta(v)
		phi_ret = geo.spherical_phi(v)
		assert_almost_eq(theta, theta_ret)
		assert_almost_eq(phi, phi_ret)

	@pytest.mark.parametrize("vec", testdata['vector'])
	@pytest.mark.parametrize("norm", testdata['normal'])
	@pytest.mark.parametrize("pnt", testdata['point'])
	def test_vector_normal_point_conversion(self, vec, norm, pnt):

		v = geo.Vector.from_arr(vec)
		assert_triple_eq(v, vec)
		assert_elem_eq(v, vec)
		v = geo.Vector.from_arr(norm)
		assert_elem_eq(v, norm)
		v = geo.Vector.from_arr(pnt)
		assert_elem_eq(v, pnt)

		n = geo.Normal.from_arr(vec)
		assert_elem_eq(n, vec)
		n = geo.Normal.from_arr(norm)
		assert_elem_eq(n, norm)
		assert_triple_eq(n, norm)
		n = geo.Normal.from_arr(pnt)
		assert_elem_eq(n, pnt)

		p = geo.Point.from_arr(vec)
		assert_elem_eq(p, vec)
		p = geo.Point.from_arr(norm)
		assert_elem_eq(p, norm)
		p = geo.Point.from_arr(pnt)
		assert_elem_eq(p, pnt)
		assert_triple_eq(p, pnt)

	@pytest.mark.parametrize("p1", testdata['point'])
	@pytest.mark.parametrize("p2", testdata['point'])
	def test_point_vector_arithmetic(self, p1, p2):
		v = p1 - p2
		assert isinstance(v, geo.Vector)
		assert_almost_eq(p1.sq_dist(p2), p2.sq_dist(p1))
		assert_almost_eq(p1.sq_dist(p2), p1.dist(p2) * p2.dist(p1))
		assert_almost_eq(p1.sq_dist(p2), v.sq_length())
		assert_almost_eq(p1.dist(p2), v.length())

		assert isinstance(p1 - p2, geo.Vector)
		assert isinstance(p1 + p2, geo.Point)
		assert isinstance(p1 + v, geo.Vector)
		assert isinstance(v + p1, geo.Point)
		with pytest.raises(TypeError):
			p1 -= p2

	@pytest.mark.parametrize("vec", testdata['vector'])
	def test_vec_cross_and_length(self, vec):
		assert_almost_eq(vec.cross(vec), np.array([0., 0., 0.]))
		assert_almost_eq(vec.sq_length(), vec.x * vec.x + vec.y * vec.y * vec.z)
		assert_almost_eq(vec.sq_length(), vec.length() * vec.length())

	@pytest.mark.parametrize("norm", testdata['normal'])
	def test_norm_cross_and_length(self, norm):
		assert_almost_eq(norm.cross(norm), np.array([0., 0., 0.]))
		assert_almost_eq(norm.sq_length(), norm.x * norm.x + norm.y * norm.y * norm.z)
		assert_almost_eq(norm.sq_length(), norm.length() * norm.length())

	@pytest.mark.parametrize("pnt", testdata['point'])
	def test_norm_cross_and_length(self, pnt):
		assert_almost_eq(pnt.cross(pnt), np.array([0., 0., 0.]))
		assert_almost_eq(pnt.sq_length(), pnt.x * pnt.x + pnt.y * pnt.y * pnt.z)
		assert_almost_eq(pnt.sq_length(), pnt.length() * pnt.length())

	@pytest.mark.parametrize("norm", testdata['normal'])
	def test_normal_normalize(self, norm):
		n = norm.normalize()
		assert not id(n) == id(norm)
		assert_almost_eq(norm / norm.length(), n)

	@pytest.mark.parametrize("vec", testdata['vector'])
	@pytest.mark.parametrize("pnt", testdata['point'])
	def test_ray_from_parent(self, vec, pnt):
		r0 = geo.Ray(pnt, vec)
		r1 = geo.Ray.from_parent(vec, pnt, r0)
		assert r1.time == r0.time
		assert r1.depth == r0.depth + 1

	@pytest.mark.parametrize("vec", testdata['vector'])
	@pytest.mark.parametrize("pnt", testdata['point'])
	def test_ray_from_ray(self, vec, pnt):
		r0 = geo.Ray(pnt, vec)
		r1 = geo.Ray.from_ray(r0)
		assert r1.time == r0.time
		assert r1.depth == r0.depth
		assert not id(r1.o) == id(r0.o)
		assert not id(r1.d) == id(r1.d)

	@pytest.mark.parametrize("vec", testdata['vector'])
	@pytest.mark.parametrize("pnt", testdata['point'])
	def test_ray_call(self, vec, pnt):
		ray = geo.Ray(pnt, vec)
		t = rng()
		p = ray(t)
		assert isinstance(pnt, geo.Point)
		assert p == pnt + vec * t

	@pytest.mark.parametrize("vec", testdata['vector'])
	@pytest.mark.parametrize("pnt", testdata['point'])
	def test_ray_differential_from_parent(self, vec, pnt):
		r0 = geo.RayDifferential(pnt, vec)
		r1 = geo.RayDifferential.from_parent(vec, pnt, r0)
		assert r1.time == r0.time
		assert r1.depth == r0.depth + 1

	@pytest.mark.parametrize("vec", testdata['vector'])
	@pytest.mark.parametrize("pnt", testdata['point'])
	def test_ray_differential_from_ray(self, vec, pnt):
		r0 = geo.Ray(pnt, vec)
		r1 = geo.RayDifferential.from_ray(r0)
		assert r1.time == r0.time
		assert r1.depth == r0.depth
		assert not id(r1.o) == id(r0.o)
		assert not id(r1.d) == id(r1.d)

	@pytest.mark.parametrize("vec", testdata['vector'])
	@pytest.mark.parametrize("pnt", testdata['point'])
	def test_ray_differential_from_ray_differential(self, vec, pnt):
		r0 = geo.RayDifferential(pnt, vec)
		r1 = geo.RayDifferential.from_rd(r0)
		assert r1.time == r0.time
		assert r1.depth == r0.depth
		assert r1.has_differentials == r0.has_differentials
		assert not id(r1.o) == id(r0.o)
		assert not id(r1.d) == id(r1.d)
		assert not id(r1.rxDirection) == id(r0.rxDirection)
		assert not id(r1.ryDirection) == id(r0.ryDirection)
		assert not id(r1.rxOrigin) == id(r0.rxOrigin)
		assert not id(r1.ryOrigin) == id(r0.ryOrigin)

	@pytest.mark.parametrize("vec", testdata['vector'])
	@pytest.mark.parametrize("pnt", testdata['point'])
	def test_ray_differential_call(self, vec, pnt):
		ray = geo.RayDifferential(pnt, vec)
		t = rng()
		p = ray(t)
		assert isinstance(pnt, geo.Point)
		assert p == pnt + vec * t

	@pytest.mark.parametrize("vec", testdata['vector'])
	@pytest.mark.parametrize("pnt", testdata['point'])
	def test_ray_differential_scale(self, vec, pnt):
		ray = geo.RayDifferential(pnt, vec)
		ray.rxOrigin = ray.ryOrigin = pnt
		ray.rxDirection = vec + rng(3)
		ray.ryDirection = vec + rng(3)
		s = rng()
		ray.scale_differential(s)
		assert ray.rxOrigin == ray.o + (ray.rxOrigin - ray.o) * s
		assert ray.ryOrigin == ray.o + (ray.ryOrigin - ray.o) * s
		assert ray.rxDirection == ray.d + (ray.rxDirection - ray.d) * s
		assert ray.ryDirection == ray.d + (ray.ryDirection - ray.d) * s
		assert isinstance(ray.rxOrigin, geo.Point)
		assert isinstance(ray.ryOrigin, geo.Point)
		assert isinstance(ray.rxDirection, geo.Vector)
		assert isinstance(ray.ryDirection, geo.Vector)

	@pytest.mark.parametrize("p1", testdata['point'])
	def test_bbox(self, p1):
		var = 5.

		b = geo.BBox()
		assert b.pMin == geo.Point(np.inf, np.inf, np.inf)
		assert b.pMax == geo.Point(-np.inf, -np.inf, -np.inf)

		b = geo.BBox(p1=None, p2=p1)
		assert b.pMin == geo.Point(-np.inf, -np.inf, -np.inf)
		assert b.pMax == p1
		assert not id(b.Max) == id(p1)

		geo.BBox(p1=p1, p2=None)
		assert b.pMax == geo.Point(np.inf, np.inf, np.inf)
		assert b.pMin == p1
		assert not id(b.Min) == id(p1)

		p0 = geo.Point(0., 0., 0.)
		p2 = (p1 + rng(3) * var).view(geo.Point)

		b = geo.BBox(p1, p0)
		assert b.pMin == p0
		assert b.pMin == p1
		assert not id(b.pMin) == id(p0)
		assert not id(b.pMax) == id(p1)

		b1 = geo.BBox(p0, p1)
		b2 = geo.BBox.from_bbox(b1)
		assert not id(b1.pMax) == id(b2.pMax)
		assert not id(b1.pMin) == id(b2.pMin)
		assert b1 == b2

		b2 = geo.BBox(p0, p2)
		assert b1 != b2

		b3 = geo.BBox(p1, p2)
		b4 = geo.BBox.union(b1, b3)
		assert b4 == b2

		b4 = geo.BBox.union(b1, p2)
		assert b4 == b2

		b5 = geo.BBox.from_bbox(b1)
		b5.union(b3)
		assert b5 == b4

		b5 = geo.BBox.from_bbox(b1)
		b5.union(p2)
		assert b5 == b4

		assert b5.overlaps(b1)
		assert b5.overlaps(b2)
		assert b5.overlaps(b3)

		assert b5.inside(p1)

		assert not b1.inside(p2)
		b1.expand(var)
		assert b1.inside(p2)

	def test_bbox_area_vol(self):
		b = geo.BBox(geo.Point(0., 0., 0.), geo.Point(2., 2., 2.))
		assert b.surface_area() == 24.
		assert b.volume() == 8.

	def test_bbox_extent(self):
		for i in range(3):
			p = geo.Point(1., 1., 1.,)
			p[i] = 2.
			b = geo.BBox(geo.Point(0., 0., 0.), p)
			assert b.maximum_extent() == i

	def test_bbox_bounding_sphere(self):
		b = geo.BBox(geo.Point(0., 0., 0.), geo.Point(1., 1., 1.))
		ctr, rad = b.bounding_sphere()
		assert ctr == geo.Point(.5, .5, .5)
		assert_almost_eq(rad, np.sqrt(3) / 2)

	def test_bbox_lerp(self):
		b = geo.BBox(geo.Point(0., 0., 0.), geo.Point(1., 1., 1.))
		tx, ty, tz = rng(3)
		p = b.lerp(tx, ty, tz)
		assert_almost_eq(p.sq_length(), tx * tx + ty * ty + tz * tz)

	def test_bbox_offset(self):
		b = geo.BBox(geo.Point(0., 0., 0.), geo.Point(1., 1., 1.))
		tx, ty, tz = rng(3)
		p = b.lerp(tx, ty, tz)
		v = b.offset(p)
		assert_almost_eq(v, [tx, ty, tz])

	def test_bbox_intersect(self):
		b = geo.BBox(geo.Point(1., 1., 1.), geo.Point(2., 2., 2.))
		ray = geo.Ray(geo.Point(0., 0., 0.), geo.Vector(1., 1., 1.))
		hit, t1, t2 = b.intersect_p(ray)
		assert hit
		assert_almost_eq(t1, 1.)
		assert_almost_eq(t2, 2.)

		ray = geo.Ray(geo.Point(1., 1., 0.), geo.Vector(0., 0., 1.))
		with pytest.raises(RuntimeWarning):
			hit, t1, t2 = b.intersect_p(ray)
			assert hit
			assert_almost_eq(t1, 1.)
			assert_almost_eq(t2, 2.)

		ray = geo.Ray(geo.Point(1., 1., 1.), geo.Vector(-1., -1., -1.))
		hit, t1, t2 = b.intersect_p(ray)
		assert hit
		assert_almost_eq(t1, 0.)
		assert_almost_eq(t2, 0.)

		ray = geo.Ray(geo.Point(0., 0., 0.), geo.Vector(-1., -1., -1.))
		hit, t1, t2 = b.intersect_p(ray)
		assert not hit
		assert_almost_eq(t1, 0.)
		assert_almost_eq(t2, 0.)
