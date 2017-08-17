"""
measured.py

pytracer.reflection.bdf package

Measured BRDF. Support brdf and MERL protocol.

TODO: Read MERL binaries.


Created by Jiayao on Aug 2, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo
from pytracer.reflection.utility import (cos_phi, sin_phi, cos_theta, sin_theta, brdf_remap)
from pytracer.reflection.bdf.bdf import (BDFType, BDF)


__all__ = ['IrIsotropicBRDFSample', 'IrIsotropicBRDF', 'ReHalfangleBRDF']


class IrIsotropicBRDFSample(object):
	"""
	IrIsotropicBRDFSample Class

	Store irregular isotropic BRDF sample
	"""

	def __init__(self, p: 'geo.Point', v: 'Spectrum'):
		self.p = p.copy()
		self.v = v.copy()

	def __repr__(self):
		return "{}\ngeo.Point: {}\nSpectrum:\n{}".format(self.__class__,
		                                                 self.p, self.v)


class IrIsotropicBRDF(BDF):
	"""
	IrIsotropicBRDF Class

	Used for measured BRDF
	"""

	def __init__(self, tree: 'KdTree', data: ['IrIsotropicBRDFSample']):
		super().__init__(BDFType.REFLECTION | BDFType.GLOSSY)
		self.tree = tree
		self.data = data

	def f(self, wo: 'geo.Vector', wi: 'geo.Vector') -> 'Spectrum':
		m = brdf_remap(wo, wi)
		max_dist = .01
		while True:
			idx = self.tree.query_ball_point(m, max_dist)
			if len(idx) > 2 or max_dist > 1.5:
				sum_wt = 0.
				v = Spectrum(0.)
				for sample in self.data[idx]:
					wt = np.exp((sample.p - m).sq_length() * -100.)
					sum_wt += wt
					v += sample.v * wt

				return v.clip() / sum_wt

			max_dist *= 1.414


class ReHalfangleBRDF(BDF):
	"""
	ReHalfangleBRDF Class

	Models regularly tabulated BRDF Samples
	Format inline with Matusik(2003) at
	http://people.csail.mit.edu/wojciech/BRDFDatabase/
	"""

	def __init__(self, data: 'np.ndarray', nth: INT, ntd: INT, npd: INT):
		super().__init__(BDFType.REFLECTION | BDFType.GLOSSY)
		self.brdf = data
		self.nThetaH = INT(nth)
		self.nThetaD = INT(ntd)
		self.nPhiD = INT(npd)

	def f(self, wo: 'geo.Vector', wi: 'geo.Vector') -> 'Spectrum':
		# find wh and transform wi to halfangle coord. system
		wh = geo.normalize(wi + wo)

		t = geo.spherical_theta(wh)
		cp = cos_phi(wh)
		sp = sin_phi(wh)
		ct = cos_theta(wh)
		st = sin_theta(wh)
		x = geo.Vector(cp * ct, sp * ct, -st)
		y = geo.Vector(-sp, cp, 0.)
		wd = geo.Vector(wi.dot(x), wi.dot(y), wi.dot(wh))

		# compute index
		wdt = geo.spherical_theta(wd)
		wdp = geo.spherical_phi(wd)
		if wdp > PI:
			wdp -= PI
		wht_idx = np.clip(INT((np.sqrt(max(0., t / (PI / 2.)))) / self.nThetaH), 0, self.nThetaH - 1)

		wdt_idx = np.clip(INT(wdt / (self.nThetaD * PI)), 0, self.nThetaD - 1)

		wdp_idx = np.clip(INT(wdp / (self.nPhiD * PI)), 0, self.nPhiD - 1)
		return Spectrum.from_rgb(self.brdf[wht_idx][wdt_idx][wdp_idx])
