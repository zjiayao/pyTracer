"""
utility.py

pytracer.sampler package

Utility functions for
samplers.

Created by Jiayao on Aug 9, 2017
"""
from __future__ import absolute_import
from pytracer import *

__all__ = ['stratified_sample_1d', 'stratified_sample_2d', 'latin_hypercube_1d',
           'latin_hypercube_2d']


def stratified_sample_1d(nSamples: INT, jitter: bool=True, rng=np.random.rand) -> 'np.ndarray':
	if not jitter:
		return (np.arange(nSamples) + .5) / nSamples
	else:
		return (np.arange(nSamples) + rng(nSamples)) / nSamples


def stratified_sample_2d(nx: INT, ny: INT, jitter: bool=True, rng=np.random.rand) -> 'np.ndarray':

	if jitter:
		return np.column_stack([(np.tile(np.arange(nx),   ny) + rng(nx * ny)) / nx,
								(np.repeat(np.arange(ny), nx) + rng(nx * ny)) / ny])
	else:
		return np.column_stack([(np.tile(np.arange(nx),   ny) + .5) / nx,
								(np.repeat(np.arange(ny), nx) + .5) / ny])


def latin_hypercube_1d(nSamples: INT, rng=np.random.rand) -> 'np.ndarray':
	ret = (np.arange(nSamples) + rng(nSamples)) / nSamples
	np.random.shuffle(ret)
	return ret


def latin_hypercube_2d(n: INT, rng=np.random.rand) -> 'np.ndarray':
	ys = (np.arange(n) + rng(n)) / n
	np.random.shuffle(ys)
	return np.column_stack([(np.arange(n) + rng(n)) / n, ys])