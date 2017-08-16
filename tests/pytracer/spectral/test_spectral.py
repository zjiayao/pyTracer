"""
test_spectral.py

A test script that (roughly) test
the implementation of various classes

Created by Jiayao on 15 Aug, 2017
"""
from __future__ import absolute_import

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from pytracer.spectral.spectrum import (xyz2rgb, rgb2xyz, SpectrumType,
                                        CoefficientSpectrum, SampledSpectrum,
                                        RGBSpectrum)

N_TEST_CASE = 10
EPS = 5
VAR = 10
np.random.seed(1)
rng = np.random.rand
rng_uniform = np.random.uniform

test_data = {
	'triple': [np.array([rng_uniform(0., 1.), rng_uniform(0., 1.), rng_uniform(0., 1.)]) for _ in range(N_TEST_CASE)],
	'type': [SpectrumType.REFLECTANCE, SpectrumType.ILLUMINANT]

}


@pytest.fixture
def lambda_smp():
	return 350 + rng(50) * 900


@pytest.fixture
def value_smp():
	return 20 + rng(50) * 10


def assert_almost_eq(a, b, thres=EPS):
	assert_array_almost_equal(a, b, thres)


class TestUtil(object):

	@pytest.mark.parametrize("arr", test_data['triple'])
	def test_xyz_rgb(self, arr):
		rgb = xyz2rgb(arr)
		xyz = rgb2xyz(rgb)
		assert_almost_eq(xyz, arr)

		xyz = rgb2xyz(arr)
		rgb = xyz2rgb(xyz)
		assert_almost_eq(rgb, arr)


class TestCoefSpec(object):
	def test_init_and_copy(self):
		cs = CoefficientSpectrum(10)
		assert np.shape(cs.c)[0] == 10
		assert cs == [0.] * 10

		cs = CoefficientSpectrum(10, 1.)
		assert np.shape(cs.c)[0] == 10
		assert cs == [1.] * 10

		css = cs.copy()
		assert cs == css
		assert not cs != css

	@pytest.mark.parametrize("arr", test_data['triple'])
	def test_create(self, arr):
		cs = CoefficientSpectrum.create(arr)
		assert cs.c == arr

		cs = CoefficientSpectrum.create(list(arr))
		assert cs.c == arr

	def test_is_black(self):
		cs = CoefficientSpectrum(10)
		assert cs.is_black()
		cs.c[0] = 1
		assert not cs.is_black()

	@pytest.mark.parametrize("arr", test_data['triple'])
	def test_arith(self, arr):
		cs = CoefficientSpectrum.create(arr)
		assert_almost_eq(cs.sqrt().c, np.sqrt(cs.c))
		assert_almost_eq(cs.exp().c, np.exp(cs.c))
		e = rng() * VAR
		assert_almost_eq(cs.pow(e).c, np.power(cs.c, e))

	@pytest.mark.parametrize("arr_1", test_data['triple'])
	@pytest.mark.parametrize("arr_2", test_data['triple'])
	def test_lerp(self, arr_1, arr_2):
		cs_1 = CoefficientSpectrum.create(arr_1)
		cs_2 = CoefficientSpectrum.create(arr_2)
		t = rng()
		cs = cs_1.lerp(t, cs_2)
		c = (1. - t) * cs_1.c + t * cs_2.c
		assert_almost_eq(cs.c, c)


class TestSampledAndRGBSpec(object):
	@classmethod
	def setup_class(cls):
		SampledSpectrum.init()

	@pytest.mark.parametrize("arr", test_data['triple'])
	def test_init(self, arr):
		from pytracer.spectral.spectrum import N_SPECTRAL_SAMPLES
		ss = SampledSpectrum()
		assert ss.n_samples == N_SPECTRAL_SAMPLES
		assert ss == [0.] * N_SPECTRAL_SAMPLES
		ss = SampledSpectrum(arr)
		assert ss.n_samples == 3
		assert ss == arr

		rs = RGBSpectrum()
		assert rs.n_samples == 3
		assert rs == [0.] * 3
		rs = RGBSpectrum(arr)
		assert rs.n_samples == 3
		assert rs == arr

	def test_from_sampled(self, lambda_smp, value_smp):
		ss = SampledSpectrum.from_sampled(lambda_smp, value_smp)
		rs = RGBSpectrum.from_sampled(lambda_smp, value_smp)
		assert_almost_eq(ss.to_rgb(), rs)

	def test_avg_spec_smp(self):
		assert_almost_eq(SampledSpectrum.average_spectrum_samples([1, 2, 3, 4], [3, 4, 5, 3], 1.5, 3.5), 4.3125)
		assert_almost_eq(SampledSpectrum.average_spectrum_samples([1, 2, 3, 4], [3, 4, 5, 3], 0.5, 4), 3.8571428571428571)
		assert_almost_eq(SampledSpectrum.average_spectrum_samples([1, 2, 3, 4], [3, 4, 5, 3], 0.5, 4.5), 3.75)

	# TODO: Full spectrum
	# @pytest.mark.parametrize("arr", test_data['triple'])
	# @pytest.mark.parametrize("tp", test_data['type'])
	# def test_conversion(self, arr, tp):
	# 	rgb = arr
	# 	xyz = rgb2xyz(rgb)
	#
	# 	ss = SampledSpectrum.from_rgb(rgb)
	# 	rs = RGBSpectrum(rgb)
	# 	rss = RGBSpectrum.from_xyz(xyz)
	# 	assert_almost_eq(rs.to_rgb(), rs)
	# 	assert_almost_eq(ss.to_rgb(), rs)
	# 	assert_almost_eq(ss.to_xyz(), rs.to_xyz())
	# 	assert_almost_eq(rss.to_rgb(), rss)
	# 	assert_almost_eq(ss.to_rgb(), rss)
	# 	assert_almost_eq(rss.to_xyz(), ss.to_xyz())
	#
	# 	ss = SampledSpectrum.from_rgb(rgb, tp)
	# 	rs = RGBSpectrum.from_xyz(xyz, tp)
	# 	assert_almost_eq(rs.to_rgb(), rs)
	# 	assert_almost_eq(ss.to_rgb(), rs)
	# 	assert_almost_eq(rs.to_xyz(), ss.to_xyz())
	#
	# 	sss = SampledSpectrum.from_xyz(xyz, tp)
	# 	rrs = RGBSpectrum.from_rgb(rgb, tp)
	# 	assert_almost_eq(rrs.to_rgb(), rrs)
	# 	assert_almost_eq(sss.to_rgb(), rrs)
	# 	assert_almost_eq(rrs.to_xyz(), sss.to_xyz())
	#

















