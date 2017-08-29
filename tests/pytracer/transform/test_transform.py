"""
test_transform.py

A test script that (roughly) test
the implementation of various classes

Created by Jiayao on 15 Aug, 2017
"""
from __future__ import absolute_import

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import pytracer.geometry as geo
from pytracer.transform import (Transform, AnimatedTransform)
import pytracer.transform.quat as quat

N_TEST_CASE = 5
VAR = 10.
EPS = 6
np.random.seed(1)
rng = np.random.rand

test_data = {
	'vector': [geo.Vector(rng(), rng(), rng()) * VAR for _ in range(N_TEST_CASE)],
	'normal': [geo.Normal(rng(), rng(), rng()) * VAR for _ in range(N_TEST_CASE)],
	'point': [geo.Point(rng(), rng(), rng()) * VAR for _ in range(N_TEST_CASE)],
	'quat': [quat.Quaternion(rng(), rng(), rng(), rng()) * VAR for _ in range(N_TEST_CASE)],

}
test_data['vector'].extend([geo.Vector(0., 0., 0.), geo.Vector(1., 0., 0.), geo.Vector(0., 1., 0., ), geo.Vector(1., 0., 0.)])
test_data['normal'].extend([geo.Normal(0., 0., 0.), geo.Normal(1., 0., 0.), geo.Normal(0., 1., 0., ), geo.Normal(1., 0., 0.)])
test_data['point'].extend([geo.Point(0., 0., 0.), geo.Point(1., 0., 0.), geo.Point(0., 1., 0., ), geo.Point(1., 0., 0.)])
test_data['quat'].extend([quat.Quaternion(0., 0., 0., 0.)])

geometry_data = {
	'trans_vec': [geo.Vector(1., 2., 3.)],
	'scale_coef': [geo.Vector(1., 2., 3.)],
	'origin': [geo.Point(0., 0., 0.)],
	'x_axis': [geo.Vector(1., 0., 0.)],
	'y_axis': [geo.Vector(0., 1., 0.)],
	'z_axis': [geo.Vector(0., 0., 1.)],
	't1': [Transform.look_at(geo.Point(0., 0., 0.),
	                         geo.Point(0., 0., 1.),
	                         geo.Vector(0., 1., 0.))],
	't2': [Transform.look_at(geo.Point(0., 1., 1.),
	                         geo.Point(0., 0., 1.),
	                         geo.Vector(0., 0., 1.))],
}


def assert_almost_eq(a, b, thres=EPS):
	assert a == b or a == pytest.approx(b, abs=thres)


def assert_elem_eq(a, b, thres=EPS):
	for i in range(len(a)):
		assert_almost_eq(a[i], b[i])


class TestTransform(object):

	def test_init(self):
		m = np.random.rand(4, 4)
		m_inv = np.linalg.inv(m)

		t = Transform(m, m_inv)
		assert (m == t.m).all()
		assert not id(m) == id(t.m)
		assert (m_inv == t.m_inv).all()
		assert not id(m_inv) == id(t.m_inv)

		with pytest.raises(AttributeError):
			t.m = m

		with pytest.raises(AttributeError):
			t.m_inv = m

		t = Transform()
		assert not id(t.m) == id(t.m_inv)
		assert (t.m == t.m_inv).all()
		assert (t.m == np.eye(4)).all()

		t = Transform(m)
		assert_array_almost_equal(t.m_inv, m_inv)

		with pytest.raises(TypeError):
			t = Transform(np.random.rand(3,3))

		t = Transform()
		tt = t.copy()
		assert (tt == t)
		assert not id(tt) == id(t)

	def test_inverse(self):
		t = Transform(rng(4,4))
		ti = t.inverse()
		assert_array_almost_equal(t.m, ti.m_inv)
		assert_array_almost_equal(t.m_inv, ti.m)

	def test_identity(self):
		assert Transform().is_identity()
		assert not Transform(rng(4, 4)).is_identity()

	def test_has_scale(self):
		assert not Transform().has_scale()
		assert Transform(rng(4, 4)).has_scale()

	def test_swap_handedness(self):
		assert not Transform().swaps_handedness()

	@pytest.mark.parametrize("vec", test_data['vector'])
	def test_call_vector(self, vec):
		t = Transform()
		v = t(vec)
		assert isinstance(v, geo.Vector)
		assert v == vec
		assert not id(v) == id(vec)

		v = t(np.array(vec), dtype=geo.Vector)
		assert isinstance(v, geo.Vector)
		assert v == vec
		assert not id(v) == id(vec)

	@pytest.mark.parametrize("pnt", test_data['point'])
	def test_call_point(self, pnt):
		t = Transform()
		p = t(pnt)
		assert isinstance(p, geo.Point)
		assert p == pnt
		assert not id(p) == id(pnt)

		p = t(np.array(pnt), dtype=geo.Point)
		assert isinstance(p, geo.Point)
		assert p == pnt
		assert not id(p) == id(pnt)

	@pytest.mark.parametrize("norm", test_data['normal'])
	def test_call_normal(self, norm):
		t = Transform()
		n = t(norm)
		assert isinstance(n, geo.Normal)
		assert n == norm
		assert not id(n) == id(norm)

		n = t(np.array(norm), dtype=geo.Normal)
		assert n == norm
		assert not id(n) == id(norm)

	@pytest.mark.parametrize("pnt", test_data['point'])
	@pytest.mark.parametrize("vec", test_data['vector'])
	def test_call_ray(self, pnt, vec):
		ray = geo.Ray(pnt, vec)
		t = Transform()
		r = t(ray)
		assert isinstance(r, geo.Ray)
		assert r.o == ray.o
		assert r.d == ray.d

	@pytest.mark.parametrize("pnt", test_data['point'])
	@pytest.mark.parametrize("vec", test_data['vector'])
	def test_call_ray_differential(self, pnt, vec):
		ray = geo.RayDifferential(pnt, vec)
		t = Transform()
		r = t(ray)
		assert isinstance(r, geo.RayDifferential)
		assert r.o == ray.o
		assert r.d == ray.d

	@pytest.mark.parametrize("p1", test_data['point'])
	@pytest.mark.parametrize("p2", test_data['point'])
	def test_call_bbox(self, p1, p2):
		box = geo.BBox(p1, p2)
		t = Transform()
		b = t(box)
		assert isinstance(b, geo.BBox)
		assert_elem_eq(b.pMax, box.pMax)
		assert_elem_eq(b.pMin, box.pMin)

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
		assert (t.m == m1.dot(m2)).all()
		assert (t.m_inv == m2_inv.dot(m1_inv)).all()

	# translation
	@pytest.mark.parametrize("vec", test_data['vector'])
	@pytest.mark.parametrize("trans_vec", geometry_data['trans_vec'])
	def test_translate_vec(self, vec, trans_vec):
		t = Transform.translate(trans_vec)

		v = t(vec)
		assert v == vec

		vv = t.inverse()(v)
		assert vv == vec

	@pytest.mark.parametrize("pnt", test_data['point'])
	@pytest.mark.parametrize("trans_vec", geometry_data['trans_vec'])
	def test_translate_pnt(self, pnt, trans_vec):
		t = Transform.translate(trans_vec)

		p = t(pnt)
		assert p == pnt + trans_vec

		pp = t.inverse()(p)
		assert_elem_eq(pp, pnt)

	@pytest.mark.parametrize("norm", test_data['normal'])
	@pytest.mark.parametrize("trans_vec", geometry_data['trans_vec'])
	def test_translate_norm(self, norm, trans_vec):
		t = Transform.translate(trans_vec)

		n = t(norm)
		assert n == norm

		nn = t.inverse()(n)
		assert_elem_eq(nn, norm)

	@pytest.mark.parametrize("p1", test_data['point'])
	@pytest.mark.parametrize("p2", test_data['point'])
	@pytest.mark.parametrize("trans_vec", geometry_data['trans_vec'])
	def test_translate_bbox(self, p1, p2, trans_vec):
		t = Transform.translate(trans_vec)

		b = geo.BBox(p1, p2)
		bb = t(b)

		assert_elem_eq(bb.pMin, b.pMin + trans_vec)
		assert_elem_eq(bb.pMax, b.pMax + trans_vec)

		bbb = t.inverse()(bb)
		assert_elem_eq(bbb.pMin, b.pMin)
		assert_elem_eq(bbb.pMax, b.pMax)

	# scale
	@pytest.mark.parametrize("vec", test_data['vector'])
	@pytest.mark.parametrize("scale_coef", geometry_data['scale_coef'])
	def test_scale_vec(self, vec, scale_coef):
		t = Transform.scale(scale_coef[0],
		                    scale_coef[1],
		                    scale_coef[2])

		v = t(vec)
		assert_elem_eq(v, vec * scale_coef)

		vv = t.inverse()(v)
		assert_elem_eq(vv, vec)

	@pytest.mark.parametrize("pnt", test_data['point'])
	@pytest.mark.parametrize("scale_coef", geometry_data['scale_coef'])
	def test_scale_pnt(self, pnt, scale_coef):
		t = Transform.scale(scale_coef[0],
		                    scale_coef[1],
		                    scale_coef[2])

		p = t(pnt)
		assert_elem_eq(p, pnt * scale_coef)

		pp = t.inverse()(p)
		assert_elem_eq(pp, pnt)

	@pytest.mark.parametrize("norm", test_data['normal'])
	@pytest.mark.parametrize("scale_coef", geometry_data['scale_coef'])
	def test_scale_norm(self, norm, scale_coef):
		t = Transform.scale(scale_coef[0],
		                    scale_coef[1],
		                    scale_coef[2])

		n = t(norm)
		assert_elem_eq(n, norm / scale_coef)

		nn = t.inverse()(n)
		assert_elem_eq(nn, norm)

	@pytest.mark.parametrize("p1", test_data['point'])
	@pytest.mark.parametrize("p2", test_data['point'])
	@pytest.mark.parametrize("scale_coef", geometry_data['scale_coef'])
	def test_scale_bbox(self, p1, p2, scale_coef):
		t = Transform.scale(scale_coef[0],
		                    scale_coef[1],
		                    scale_coef[2])

		b = geo.BBox(p1, p2)
		bb = t(b)

		assert_elem_eq(bb.pMin, b.pMin * scale_coef)
		assert_elem_eq(bb.pMax, b.pMax * scale_coef)

		bbb = t.inverse()(bb)
		assert_elem_eq(bbb.pMin, b.pMin)
		assert_elem_eq(bbb.pMax, b.pMax)

	@pytest.mark.parametrize("x_axis", geometry_data['x_axis'])
	@pytest.mark.parametrize("y_axis", geometry_data['y_axis'])
	@pytest.mark.parametrize("z_axis", geometry_data['z_axis'])
	def test_rotate_x(self, x_axis, y_axis, z_axis):
		t = Transform.rotate_x(45)
		assert_elem_eq(t(x_axis), x_axis)
		v = t(y_axis)
		assert_elem_eq(v, geo.Vector(0., np.sqrt(2) / 2., np.sqrt(2) / 2.))
		vv = t(v)
		assert_elem_eq(vv, z_axis)
		vv = t.inverse()(v)
		assert_elem_eq(vv, y_axis)

	@pytest.mark.parametrize("x_axis", geometry_data['x_axis'])
	@pytest.mark.parametrize("y_axis", geometry_data['y_axis'])
	@pytest.mark.parametrize("z_axis", geometry_data['z_axis'])
	def test_rotate_y(self, x_axis, y_axis, z_axis):
		t = Transform.rotate_y(30)
		assert_elem_eq(t(y_axis), y_axis)
		p = geo.Point.from_arr(z_axis)
		pp = t(t(t(p)))
		assert_elem_eq(pp, x_axis)

	@pytest.mark.parametrize("x_axis", geometry_data['x_axis'])
	@pytest.mark.parametrize("y_axis", geometry_data['y_axis'])
	@pytest.mark.parametrize("z_axis", geometry_data['z_axis'])
	def test_rotate_z(self, x_axis, y_axis, z_axis):
		t = Transform.rotate_z(90)
		assert_elem_eq(t(z_axis), z_axis)
		n = geo.Normal.from_arr(y_axis)
		nn = t(t(t(n)))
		assert_elem_eq(nn, x_axis)

	@pytest.mark.parametrize("x_axis", geometry_data['x_axis'])
	@pytest.mark.parametrize("y_axis", geometry_data['y_axis'])
	@pytest.mark.parametrize("z_axis", geometry_data['z_axis'])
	def test_rotate(self, x_axis, y_axis, z_axis):
		t = Transform.rotate(45., x_axis)
		assert_elem_eq(t(x_axis), x_axis)
		v = t(y_axis)
		assert_elem_eq(v, geo.Vector(0., np.sqrt(2) / 2., np.sqrt(2) / 2.))
		vv = t(v)
		assert_elem_eq(vv, z_axis)
		vv = t.inverse()(v)
		assert_elem_eq(vv, y_axis)

		t = Transform.rotate(30., y_axis)
		assert_elem_eq(t(y_axis), y_axis)
		p = geo.Point.from_arr(z_axis)
		pp = t(t(t(p)))
		assert_elem_eq(pp, x_axis)

		t = Transform.rotate(90., z_axis)
		assert_elem_eq(t(z_axis), z_axis)
		n = geo.Normal.from_arr(y_axis)
		nn = t(t(t(n)))
		assert_elem_eq(nn, x_axis)


	def test_rotate_bbox(self):
		t = Transform.rotate(180., geo.Vector(1., 1., 1.))
		p1 = geo.Point(0., 0., 0.)
		p2 = geo.Point(1., 1., 1.)
		box = geo.BBox(p1, p2)
		b = t(box)
		pmax = t(box.pMax)
		pmin = t(box.pMin)

		assert_elem_eq(b.pMin, pmin)
		assert_elem_eq(b.pMax, pmax)

		bb = t(b)
		pmax = t(pmax)
		pmin = t(pmin)
		assert_elem_eq(bb.pMin, pmin)
		assert_elem_eq(bb.pMax, pmax)

	@pytest.mark.parametrize("pnt", test_data['point'])
	@pytest.mark.parametrize("vec", test_data['vector'])
	@pytest.mark.parametrize("x_axis", geometry_data['x_axis'])
	@pytest.mark.parametrize("y_axis", geometry_data['y_axis'])
	@pytest.mark.parametrize("z_axis", geometry_data['z_axis'])
	@pytest.mark.parametrize("origin", geometry_data['origin'])
	def test_look_at(self, pnt, vec, x_axis, y_axis, z_axis, origin):
		t = Transform.look_at(pnt, vec, y_axis)

		assert_elem_eq(t(origin), pnt)
		assert_elem_eq(t.inverse()(pnt), origin)
		if not pnt == vec and not pnt == y_axis and not vec == y_axis:
			assert not t.has_scale()


	def test_orthographic(self):
		znear = rng()
		zfar = znear + rng() * VAR
		t = Transform.orthographic(znear, zfar)
		p = geo.Point(rng(), rng(), znear)
		assert_almost_eq(t(p).z, 0.)
		p = geo.Point(rng(), rng(), zfar)
		assert_almost_eq(t(p).z, 1.)

	def test_perspective(self):
		znear = rng() * VAR
		zfar = znear + rng() * VAR
		fov = rng() * 45. + 45.

		t = Transform.perspective(fov, znear, zfar)
		p = geo.Point(znear * rng(), znear * rng(), znear + rng() * (zfar - znear))
		pp = t(p)
		assert_almost_eq(pp.z, (p.z - znear) * zfar / ((zfar - znear) * p.z))
		assert_almost_eq(pp.x, p.x / (p.z * np.tan(np.deg2rad(.5 * fov))))
		assert_almost_eq(pp.y, p.y / (p.z * np.tan(np.deg2rad(.5 * fov))))


class TestQuat(object):

	@pytest.mark.parametrize("q1", test_data['quat'])
	@pytest.mark.parametrize("q2", test_data['quat'])
	def test_dot(self, q1, q2):
		q = quat.dot(q1, q2)
		assert_almost_eq(q, q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z)

	@pytest.mark.parametrize("q", test_data['quat'])
	def test_to_transform(self, q):
		t = quat.to_transform(q)

	@pytest.mark.parametrize("vec", test_data['vector'])
	def test_from_transform_and_arr(self, vec):
		t = Transform.rotate(rng() * 360., vec)
		q1 = quat.from_transform(t)
		q2 = quat.from_arr(t.m)
		assert q1 == q2

	@pytest.mark.parametrize("q1", test_data['quat'])
	@pytest.mark.parametrize("q2", test_data['quat'])
	def test_slerp(self, q1, q2):
		q = quat.slerp(rng(), q1, q2)
		assert isinstance(q, quat.Quaternion)


class TestAnimatedTransform(object):

	@pytest.mark.parametrize("t1", geometry_data['t1'])
	@pytest.mark.parametrize("t2", geometry_data['t2'])
	def test_init(self, t1, t2):
		tm = rng()
		at = AnimatedTransform(t1, tm, t1, tm)
		assert not at.animated
		at = AnimatedTransform(t1, tm, t1, tm + rng())
		assert not at.animated
		at = AnimatedTransform(t1, 0., t2, tm)
		assert at.animated

	@pytest.mark.parametrize("pnt", test_data['point'])
	@pytest.mark.parametrize("vec", test_data['vector'])
	@pytest.mark.parametrize("t1", geometry_data['t1'])
	@pytest.mark.parametrize("t2", geometry_data['t2'])
	def test_call(self, pnt, vec, t1, t2):
		ray = geo.Ray(pnt, vec)
		tm = rng()
		at = AnimatedTransform(t1, tm, t1, tm)
		assert at(ray).o == t1(ray).o
		assert at(ray).d == t1(ray).d
		assert at(rng(), pnt) == t1(pnt)
		assert at(rng(), vec) == t1(vec)

		at = AnimatedTransform(t1, 0., t2, tm)
		at(ray)
		at(rng(), pnt)
		at(rng(), vec)

	@pytest.mark.parametrize("t1", geometry_data['t1'])
	@pytest.mark.parametrize("t2", geometry_data['t2'])
	def test_motion_bounds(self, t1, t2):
		tm = rng()
		at = AnimatedTransform(t1, 0., t2, tm)
		b1 = at.motion_bounds(geo.BBox(geo.Point(0., 0., 0.), geo.Point(0., 0., 0.)), False)
		b2 = geo.BBox(geo.Point(0., 0., 0.), geo.Point(0., 1., 1.))
		assert b1 == b2

	@pytest.mark.parametrize("t1", geometry_data['t1'])
	@pytest.mark.parametrize("t2", geometry_data['t2'])
	def test_interpolate(self, t1, t2):
		tm = rng()
		at = AnimatedTransform(t1, 0., t2, tm)
		t = at.interpolate(rng())

	def test_decompose(self):
		with pytest.raises(TypeError):
			t, r, s = AnimatedTransform.decompose(rng(3, 3))
		t, r, s = AnimatedTransform.decompose(rng(4, 4))
		assert isinstance(t, geo.Vector)
		assert isinstance(r, quat.Quaternion)
		assert isinstance(s, np.ndarray)



