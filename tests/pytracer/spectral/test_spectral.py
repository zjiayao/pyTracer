"""
test_spectral.py

A test script that (roughly) test
the implementation of various classes

Created by Jiayao on 15 Aug, 2017
"""
from __future__ import absolute_import

import numpy as np
import pytest
from pytracer import EPS
from pytracer.spectral.spectrum import (xyz2rgb, rgb2xyz, SpectrumType,
                                        CoefficientSpectrum, SampledSpectrum,
                                        RGBSpectrum)

N_TEST_CASE = 5
VAR = 10
np.random.seed(1)
rng = np.random.rand

testdata = {
	'triple': [np.array([rng(), rng(), rng()]) for _ in range(N_TEST_CASE)]
}


def assert_almost_eq(a, b, thres=EPS):
	assert a == b or a == pytest.approx(b, abs=thres)


def assert_triple_eq(a: 'np.ndarray', b: 'np.ndarray', thres=EPS):
	assert a == b or a == pytest.approx(b, abs=thres)


def assert_elem_eq(a: 'np.ndarray', b: 'np.ndarray', thres=EPS):
	assert np.shape(a) == np.shape(b)
	for i in range(len(a)):
		assert_almost_eq(a[i], b[i])

# @pytest.mark.parametrize("pnt", test_data['point'])

