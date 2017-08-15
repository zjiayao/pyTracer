"""
quat.py

This module is part of the pyTracer, which
defines `Quaternion` classes.

v0.0
Created by Jiayao on July 30, 2017
"""
from __future__ import absolute_import
from .. import *
import numpy
import quaternion
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from pytracer.transform import Transform

Quaternion = numpy.quaternion


def dot(q1: 'Quaternion', q2: 'Quaternion') -> 'FLOAT':
	return q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z


def to_transform(q: 'Quaternion') -> 'Transform':
	x, y, z, w = q.w, q.x, q.y, q.z
	m = np.array([[1. - 2. * (y * y + z * z), 2. * (x * y + z * w), 2. * (x * z - y * w), 0.],
				  [2. * (x * y - z * w), 1. - 2. * (x * x + z * z), 2. * (y * z + x * w), 0.],
				  [2. * (x * z + y * w), 2. * (y * z - x * w), 1. - 2. * (x * x + y * y), 0.],
	              [0., 0., 0., 1.]],
				  dtype=FLOAT)
	from pytracer.transform import Transform
	return Transform(m.T, m)    # transpose for left-handedness


def from_transform(t: 'Transform') -> 'Quaternion':
	m = t.m
	tr = m.trace()

	if tr > .0:
		s = np.sqrt(tr + 1.)
		w = s / 2.
		s = .5 / s
		x = (m[2, 1] - m[1, 2]) * s
		y = (m[0, 2] - m[2, 0]) * s
		z = (m[1, 0] - m[0, 1]) * s
		return Quaternion(x, y, z, w)

	else:
		nxt = [1, 2, 0]
		q = [0, 0, 0]
		i = 0
		if m[1, 1] > m[0, 0]:
			i = 1
		if m[2, 2] < m[i, i]:
			i = 2
		j = nxt[i]
		k = nxt[j]

		s = np.sqrt((m[i, i] - (m[j, j] + m[k, k])) + 1.)

		q[i] = s * .5
		if s != .0:
			s = .5 / s
		w = (m[k, j] - m[j, k]) * s
		q[j] = (m[j, i] + m[i, j]) * s
		q[k] = (m[k, i] + m[i, k]) * s

		return Quaternion(q[0], q[1], q[2], w)


def from_arr(t: 'np.ndarray') -> 'Quaternion':
	m = t
	tr = m.trace()

	if tr > .0:
		s = np.sqrt(tr + 1.)
		w = s / 2.
		s = .5 / s
		x = (m[2, 1] - m[1, 2]) * s
		y = (m[0, 2] - m[2, 0]) * s
		z = (m[1, 0] - m[0, 1]) * s
		return Quaternion(x, y, z, w)

	else:
		nxt = [1, 2, 0]
		q = [0, 0, 0]
		i = 0
		if m[1, 1] > m[0, 0]:
			i = 1
		if m[2, 2] < m[i, i]:
			i = 2
		j = nxt[i]
		k = nxt[j]

		s = np.sqrt((m[i, i] - (m[j, j] + m[k, k])) + 1.)

		q[i] = s * .5
		if s != .0:
			s = .5 / s
		w = (m[k, j] - m[j, k]) * s
		q[j] = (m[j, i] + m[i, j]) * s
		q[k] = (m[k, i] + m[i, k]) * s

		return Quaternion(q[0], q[1], q[2], w)


def slerp(t: FLOAT, q1: 'Quaternion', q2: 'Quaternion') -> 'Quaternion':
	cos_theta = dot(q1, q2)
	if (cos_theta > 1. - EPS):
		return ((1. - t) * q1 + t * q2).normalized()
	else:
		theta = np.arccos(np.clip(cos_theta, -1., 1.))
		thetap = theta * t
		qperp = (q2 - q1 * cos_theta).normalized()
		return q1 * np.cos(thetap) + qperp * np.sin(thetap)

