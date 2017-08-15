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
from pytracer.transform import (Transform, AnimatedTransform)
import pytracer.transform.quat as quat

N_TEST_CASE = 5
VAR = 10
np.random.seed(1)
rng = np.random.rand

testdata = {
	'vector': [geo.Vector(rng(), rng(), rng()) * VAR for _ in range(N_TEST_CASE)],
	'normal': [geo.Normal(rng(), rng(), rng()) * VAR for _ in range(N_TEST_CASE)],
	'point': [geo.Point(rng(), rng(), rng()) * VAR for _ in range(N_TEST_CASE)],
	'quat': [quat.Quaternion(rng(), rng(), rng(), rng()) * VAR for _ in range(N_TEST_CASE)],

}
testdata['vector'].extend([geo.Vector(0., 0., 0.), geo.Vector(1., 0., 0.), geo.Vector(0., 1., 0.,), geo.Vector(1., 0., 0.)])
testdata['normal'].extend([geo.Normal(0., 0., 0.), geo.Normal(1., 0., 0.), geo.Normal(0., 1., 0.,), geo.Normal(1., 0., 0.)])
testdata['point'].extend([geo.Point(0., 0., 0.), geo.Point(1., 0., 0.), geo.Point(0., 1., 0.,), geo.Point(1., 0., 0.)])
testdata['quat'].extend([quat.Quaternion(0., 0., 0., 0.)])


def assert_almost_eq(a, b, thres=EPS):
	assert a == b or a == pytest.approx(b, abs=thres)


def assert_triple_eq(a: 'np.ndarray', b: 'np.ndarray', thres=EPS):
	assert a == b or a == pytest.approx(b, abs=thres)


def assert_elem_eq(a: 'np.ndarray', b: 'np.ndarray', thres=EPS):
	assert np.shape(a) == np.shape(b)
	for i in range(len(a)):
		assert_almost_eq(a[i], b[i])


class TestTransform(object):

	def __init__(self):
		self.trans_vec = geo.Vector(1., 2., 3.)
		self.scale_coef = np.array([1., 2., 3.])
		self.origin = geo.Point(0., 0., 0.)
		self.x_axis = geo.Vector(1., 0., 0.)
		self.y_axis = geo.Vector(0., 1., 0.)
		self.z_axis = geo.Vector(0., 0., 1.)

	def test_init(self):
		m = np.random.rand(4, 4)
		m_inv = np.linalg.inv(m)

		t = Transform(m, m_inv)
		assert m == t.m
		assert not id(m) == id(t.m)
		assert m_inv == t.m_inv
		assert not id(m_inv) == id(t.m_inv)

		with pytest.raises(PermissionError):
			t.m = m

		with pytest.raises(PermissionError):
			t.m_inv = m

		t = Transform()
		assert not id(t.m) == id(t.m_inv)
		assert t.m == t.m_inv
		assert t.m == np.eye(4)

		t = Transform(m)
		assert_almost_eq(t.m_inv, m_inv)

		with pytest.raises(TypeError):
			t = Transform(np.random.rand(3,3))

		t = Transform()
		tt = t.copy()
		assert tt == t
		assert not id(tt) == id(t)

	def test_inverse(self):
		t = Transform(rng(4,4))
		ti = t.inverse()
		assert_almost_eq(t.m, ti.m_inv)
		assert_almost_eq(t.m_inv, ti.m)

	def test_identity(self):
		assert Transform().is_identity()
		assert not Transform(rng(4, 4)).is_identity()

	def test_has_scale(self):
		assert not Transform().has_scale()
		assert Transform(rng(4, 4)).has_scale()

	def test_swap_handedness(self):
		assert not Transform().swaps_handedness()

	@pytest.mark.parametrize("vec", testdata['vector'])
	def test_call_vector(self, vec):
		t = Transform()
		v = t(vec)
		assert isinstance(v, geo.Vector)
		assert v == vec
		assert not id(v) == id(vec)

		v = t(np.ndarray(vec), dtype=geo.Vector)
		assert isinstance(v, geo.Vector)
		assert v == vec
		assert not id(v) == id(vec)

	@pytest.mark.parametrize("pnt", testdata['point'])
	def test_call_point(self, pnt):
		t = Transform()
		p = t(pnt)
		assert isinstance(p, geo.Point)
		assert p == pnt
		assert not id(p) == id(pnt)

		p = t(np.ndarray(pnt), dtype=geo.Point)
		assert isinstance(p, geo.Point)
		assert p == pnt
		assert not id(p) == id(pnt)

	@pytest.mark.parametrize("norm", testdata['normal'])
	def test_call_normal(self, norm):
		t = Transform()
		n = t(norm)
		assert isinstance(n, geo.Normal)
		assert n == norm
		assert not id(n) == id(norm)

		n = t(np.ndarray(norm), dtype=geo.Normal)
		assert n == norm
		assert not id(n) == id(norm)

	@pytest.mark.parametrize("pnt", testdata['point'])
	@pytest.mark.parametrize("vec", testdata['vector'])
	def test_call_ray(self, pnt, vec):
		ray = geo.Ray(pnt, vec)
		t = Transform()
		r = t(ray)
		assert isinstance(r, geo.Ray)
		assert r.o == ray.o
		assert r.d == ray.d

	@pytest.mark.parametrize("pnt", testdata['point'])
	@pytest.mark.parametrize("vec", testdata['vector'])
	def test_call_ray_differential(self, pnt, vec):
		ray = geo.RayDifferential(pnt, vec)
		t = Transform()
		r = t(ray)
		assert isinstance(r, geo.RayDifferential)
		assert r.o == ray.o
		assert r.d == ray.d

	@pytest.mark.parametrize("p1", testdata['point'])
	@pytest.mark.parametrize("p2", testdata['point'])
	def test_call_bbox(self, p1, p2):
		box = geo.BBox(p1, p2)
		t = Transform()
		b = t(box)
		assert isinstance(b, geo.BBox)
		assert b.pMax == box.pMax
		assert b.pMin == box.pMin

	def test_call_logic(self):
		t = Transform()
		with pytest.raises(TypeError):
			t(1.)

	def test_mul(self):
		m1 = rng(4,4)
		m2 = rng(4,4)
		m1_inv = np.linalg.inv(m1)
		m2_inv = np.linalg.inv(m2)

		t1 = Transform(m1, m1_inv)
		t2 = Transform(m2, m2_inv)

		t = t1 * t2
		assert t.m == m1.dot(m2)
		assert t.m_inv == m2_inv.dot(m1_inv)

	# translation
	@pytest.mark.parametrize("vec", testdata['vector'])
	def test_translate_vec(self, vec):
		t = Transform.translate(self.trans_vec)

		v = t(vec)
		assert v == vec

		vv = t.inverse()(v)
		assert vv == vec

	@pytest.mark.parametrize("pnt", testdata['point'])
	def test_translate_pnt(self, pnt):
		t = Transform.translate(self.trans_vec)

		p = t(pnt)
		assert p == pnt + self.trans_vec

		pp = t.inverse()(p)
		assert pp == pnt

	@pytest.mark.parametrize("norm", testdata['normal'])
	def test_translate_norm(self, norm):
		t = Transform.translate(self.trans_vec)

		n = t(norm)
		assert n == norm

		nn = t.inverse()(n)
		assert nn == norm

	@pytest.mark.parametrize("p1", testdata['point'])
	@pytest.mark.parametrize("p2", testdata['point'])
	def test_translate_bbox(self, p1, p2):
		t = Transform.translate(self.trans_vec)

		b = geo.BBox(p1, p2)
		bb = t(b)

		assert bb.pMin == b.pMin + self.trans_vec
		assert bb.pMax == b.pMax + self.trans_vec

		bbb = t.inverse()(bb)
		assert bbb.pMin == b.pMin
		assert bbb.pMax == b.pMax

	# scale
	@pytest.mark.parametrize("vec", testdata['vector'])
	def test_scale_vec(self, vec):
		t = Transform.scale(self.scale_coef[0],
		                    self.scale_coef[1],
		                    self.scale_coef[2])

		v = t(vec)
		assert_almost_eq(v, vec * self.scale_coef)

		vv = t.inverse()(v)
		assert_almost_eq(vv, vec)

	@pytest.mark.parametrize("pnt", testdata['point'])
	def test_scale_pnt(self, pnt):
		t = Transform.scale(self.scale_coef[0],
		                    self.scale_coef[1],
		                    self.scale_coef[2])

		p = t(pnt)
		assert_almost_eq(p, pnt * self.scale_coef)

		pp = t.inverse()(p)
		assert_almost_eq(pp, pnt)

	@pytest.mark.parametrize("norm", testdata['normal'])
	def test_scale_norm(self, norm):
		t = Transform.scale(self.scale_coef[0],
		                    self.scale_coef[1],
		                    self.scale_coef[2])

		n = t(norm)
		assert_almost_eq(n, norm / self.scale_coef)

		nn = t.inverse()(n)
		assert_almost_eq(nn, norm)

	@pytest.mark.parametrize("p1", testdata['point'])
	@pytest.mark.parametrize("p2", testdata['point'])
	def test_scale_bbox(self, p1, p2):
		t = Transform.scale(self.scale_coef[0],
		                    self.scale_coef[1],
		                    self.scale_coef[2])

		b = geo.BBox(p1, p2)
		bb = t(b)

		assert bb.pMin == b.pMin + geo.Vector(1., 2., 3.)
		assert bb.pMax == b.pMax + geo.Vector(1., 2., 3.)

		bbb = t.inverse()(bb)
		assert bbb.pMin == b.pMin
		assert bbb.pMax == b.pMax

	def test_rotate_x(self):
		t = Transform.rotate_x(45)
		assert_almost_eq(t(self.x_axis), self.x_axis)
		v = t(self.y_axis)
		assert_almost_eq(v, geo.Vector(0., np.sqrt(2), np.sqrt(2)))
		vv = t(v)
		assert_almost_eq(vv, self.z_axis)
		vv = t.inverse()(v)
		assert_almost_eq(vv, self.y_axis)

	def test_rotate_y(self):
		t = Transform.rotate_y(30)
		assert_almost_eq(t(self.y_axis), self.y_axis)
		p = geo.Point.from_arr(self.z_axis)
		pp = t(t(p))
		assert_almost_eq(pp, -self.x_axis)

	def test_rotate_z(self):
		t = Transform.rotate_z(90)
		assert_almost_eq(t(self.z_axis), self.z_axis)
		n = geo.Normal.from_arr(self.y_axis)
		nn = t(t(t(n)))
		assert_almost_eq(nn, self.x_axis)

	def test_rotate(self):
		t = Transform.rotate(45., self.x_axis)
		assert_almost_eq(t(self.x_axis), self.x_axis)
		v = t(self.y_axis)
		assert_almost_eq(v, geo.Vector(0., np.sqrt(2), np.sqrt(2)))
		vv = t(v)
		assert_almost_eq(vv, self.z_axis)
		vv = t.inverse()(v)
		assert_almost_eq(vv, self.y_axis)

		t = Transform.rotate(30., self.y_axis)
		assert_almost_eq(t(self.y_axis), self.y_axis)
		p = geo.Point.from_arr(self.z_axis)
		pp = t(t(p))
		assert_almost_eq(pp, -self.x_axis)

		t = Transform.rotate(90., self.z_axis)
		assert_almost_eq(t(self.z_axis), self.z_axis)
		n = geo.Normal.from_arr(self.y_axis)
		nn = t(t(t(n)))
		assert_almost_eq(nn, self.x_axis)

	@pytest.mark.parametrize("p1", testdata['point'])
	@pytest.mark.parametrize("p2", testdata['point'])
	def test_rotate_bbox(self, p1, p2):
		t = Transform.rotate(180., geo.Vector(1., 1., 1.))
		box = geo.BBox(p1, p2)
		b = t(box)

		assert_almost_eq(b.pMin, -box.pMax)
		assert_almost_eq(b.pMax, -box.pMin)

		bb = t(b)
		assert_almost_eq(bb.pMin, box.pMin)
		assert_almost_eq(bb.pmax, box.pMax)

	@pytest.mark.parametrize("pnt", testdata['point'])
	@pytest.mark.parametrize("vec", testdata['vector'])
	def test_look_at(self, pnt, vec):
		t = Transform.look_at(pnt, vec, self.y_axis)

		assert_almost_eq(t(self.origin), pnt)
		assert_almost_eq(t.inverse()(pnt), self.origin)

	def test_orthographic(self):
		znear = rng()
		zfar = znear + rng() * VAR
		t = Transform.orthographic(znear, zfar)
		p = geo.Point(rng(), rng(), znear)
		assert_almost_eq(t(p).z, 0.)
		p = geo.Point(rng(), rng(), zfar)
		assert_almost_eq(t(p).z, 1.)

	def test_perspective(self):
		znear = rng()
		zfar = znear + rng() * VAR
		fov = rng() * 90
		t = Transform.perspective(fov, znear, zfar)
		p = geo.Point(rng(), rng(), znear)
		assert_almost_eq(t(p).x, p.x)
		assert_almost_eq(t(p).y, p.y)
		assert_almost_eq(t(p).z, 0.)
		p = geo.Point(rng(), rng(), zfar)
		assert_almost_eq(t(p).x, p.x / zfar)
		assert_almost_eq(t(p).y, p.y / zfar)
		assert_almost_eq(t(p).z, 1.)


class TestQuat(object):

	@pytest.mark.parametrize("q1", testdata['quat'])
	@pytest.mark.parametrize("q2", testdata['quat'])
	def test_dot(self, q1, q2):
		q = quat.dot(q1, q2)
		assert_almost_eq(q, q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z)

	@pytest.mark.parametrize("q", testdata['quat'])
	def test_to_transform(self, q):
		t = quat.to_transform(q)

	@pytest.mark.parametrize("vec", testdata['vector'])
	def test_from_transform_and_arr(self, vec):
		t = Transform.rotate(rng() * 360., vec)
		q1 = quat.from_transform(t)
		q2 = quat.from_arr(t.m)
		assert q1 == q2

	@pytest.mark.parametrize("q1", testdata['quat'])
	@pytest.mark.parametrize("q2", testdata['quat'])
	def test_slerp(self, q1, q2):
		q = quat.slerp(rng(), q1, q2)
		assert isinstance(q, quat.Quaternion)


class TestAnimatedTransform(object):
	def __init__(self):
		self.t1 = Transform.look_at(geo.Point(0., 0., 0.),
		                            geo.Point(0., 0., 1.),
		                            geo.Vector(0., 1., 0.))
		self.t2 = Transform.look_at(geo.Point(0., 1., 1.),
		                            geo.Point(0., 0., 1.),
		                            geo.Vector(0., 0., 1.))

	def test_init(self):
		tm = rng()
		at = AnimatedTransform(self.t1, tm, self.t1, tm)
		assert not at.animated
		at = AnimatedTransform(self.t1, 0., self.t2, tm)
		assert at.animated

	@pytest.mark.parametrize("pnt", testdata['point'])
	@pytest.mark.parametrize("vec", testdata['vector'])
	def test_call(self, pnt, vec):
		ray = geo.Ray(pnt, vec)
		tm = rng()
		at = AnimatedTransform(self.t1, tm, self.t1, tm)
		assert at(ray).o == self.t1(ray).o
		assert at(ray).d == self.t1(ray).d
		assert at(rng(), pnt) == self.t1(pnt)
		assert at(rng(), vec) == self.t1(vec)

		at = AnimatedTransform(self.t1, 0., self.t2, tm)
		at(ray)
		at(rng(), pnt)
		at(rng(), vec)

	def test_motion_bounds(self):
		tm = rng()
		at = AnimatedTransform(self.t1, 0., self.t2, tm)
		b1 = at.motion_bounds()
		b2 = geo.BBox(geo.Point(0., 0., 0.), geo.Point(0., 1., 1.))
		assert b1 == b2

	def test_interpolate(self):
		tm = rng()
		at = AnimatedTransform(self.t1, 0., self.t2, tm)
		t = at.interpolate(rng())

	def test_decompose(self):
		with pytest.raises(TypeError):
			t, r, s = AnimatedTransform.decompose(rng(4, 4))
		t, r, s = AnimatedTransform.decompose(rng(4, 4))



