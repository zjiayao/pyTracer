"""
bsdf.py

pytracer.reflection package
Model BSDF, essentially a wrapper
for multiple BDFs.


Created by Jiayao on Aug 2, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo
from pytracer.reflection.bdf.bdf import (BDFType, BDF)


__all__ = ['BSDFSample', 'BSDFSampleOffset', 'BSDF']


class BSDFSample(object):
	"""
	BSDFSample Class

	Encapsulate samples used
	by BSDF.sample_f()
	"""

	def __init__(self, up0: FLOAT = 0., up1: FLOAT = 0., u_com: FLOAT = 0.):
		self.u_dir = [up0, up1]
		self.u_com = u_com

	def __repr__(self):
		return "{}\nDirection: ({},{})\nComponent: {}\n".format(self.__class__, self.u_dir[0], self.u_dir[1],
		                                                        self.u_com)

	@classmethod
	def from_rand(cls, rng=np.random.rand):
		self = cls(rng(), rng(), rng())
		return self

	@classmethod
	def from_sample(cls, sample: 'Sample', offset: 'BSDFSampleOffset', n: UINT):
		self = cls(sample.twoD[offset.offset_dir][n][0],
		           sample.twoD[offset.offset_dir][n][1],
		           sample.oneD[offset.offset_com][n])
		return self


class BSDFSampleOffset(object):
	"""
	BSDFSampleOffset Class

	Encapsulate offsets provided
	by sampler
	"""

	def __init__(self, nSamples: INT, sample: 'Sample'):
		self.nSamples = nSamples
		self.offset_com = sample.add_1d(nSamples)
		self.offset_dir = sample.add_2d(nSamples)

	def __repr__(self):
		return "{}\nSamples: {}\n".format(self.__class__, self.nSamples)


class BSDF(object):
	"""
	BSDF Class

	Models the collection of BRDF and BTDF
	Also responsible for shading Normals.

	n_s: shading normal
	n_g: geometric normal
	"""

	def __init__(self, dg: 'geo.DifferentialGeometry', ng: 'geo.Normal', e: FLOAT = 1.):
		"""
		dg: geo.DifferentialGeometry
		ng: Geometric geo.Normal
		e: index of refraction
		"""
		self.dgs = dg
		self.eta = e
		self.ng = ng

		# coord. system
		self.nn = dg.nn
		self.sn = geo.normalize(dg.dpdu)
		self.tn = self.nn.cross(self.sn)

		self.bdfs = []
		self.__nBDF = INT(0)

	def __repr__(self):
		return "{}\nBDF Count: {}\ngeo.Point: {}".format(self.__class__,
		                                                 self.nBDF, self.dgs.p)

	def sample_f(self, wo_w: 'geo.Vector', bsdf_smp: 'BSDFSample',
	             flags: 'BDFType') -> [FLOAT, 'geo.Vector', 'BDFType' 'Spectrum']:
		"""
		sample_f()

		returns [pdf, wi_w, sample_type, spectrum]
		"""
		# choose BDFs
		smp_type = None
		n_match = self.n_components(flags)
		if n_match == 0:
			return [0., None, None, Spectrum(0.)]

		cnt = last = min(util.ftoi(bsdf_smp.u_com * n_match), n_match - 1)
		for func in self.bdfs:
			if func.match_flag(flags):
				cnt -= 1
				if cnt == 0:
					bdf = func
					break
		# sample BDFs
		wo = self.w2l(wo_w)
		pdf, wi, f = bdf.sample_f(wo, bsdf_smp.u_dir[0], bsdf_smp.u_dir[1])
		wi_w = self.l2w(wi)

		if pdf == 0.:
			return [pdf, wi_w, bdf.type, Spectrum(0.)]

		# compute overall pdf
		if (not (bdf.type & BDFType.SPECULAR)) and n_match > 1:
			for func in self.bdfs:
				if func != bdf and func.match_flag(flags):
					pdf += func.pdf(wo, wi)

		if n_match > 1:
			pdf /= n_match

		# compute value of BSDF for sampled direction
		if not bdf.type & BDFType.SPECULAR:
			f = Spectrum(0.)
			if wi_w.dot(self.ng) * wo_w.dot(self.ng) > 0.:
				# ignore BTDF
				flags = BDFType(flags & ~BDFType.TRANSMISSION)
			else:
				# ignore BRDF
				flags = BDFType(flags & ~BDFType.REFLECTION)

			for func in self.bdfs:
				if func.match_flag(flags):
					f += func.f(wo, wi)

		return [pdf, wi_w, bdf.type, f]

	def pdf(self, wo: 'geo.Vector', wi: 'geo.Vector', flags: 'BDFType' = BDFType.ALL) -> FLOAT:
		if self.n_BDF == 0: return 0.
		wo = self.w2l(wo)
		wi = self.w2l(wi)
		pdf = 0.
		n_match = 0
		for bdf in self.bdfs:
			if bdf.match_flag(flags):
				n_match += 1
				pdf += bdf.pdf(wo, wi)

		return pdf / n_match if n_match > 0 else 0.

	def push_back(self, bdf: 'BDF'):
		"""
		Add more BDF
		"""
		self.bdfs.append(bdf)
		self.__nBDF += 1

	@property
	def n_BDF(self):
		return self.__nBDF

	def n_components(self, flags: 'BDFType'):
		n = 0
		for bdf in self.bdfs:
			if bdf.match_flag(flags):
				n += 1
		return n

	def w2l(self, v: 'geo.Vector') -> 'geo.Vector':
		"""
		w2l()

		Transform a `geo.Vector` in world in the local
		surface normal coord. system
		"""
		return geo.Vector(v.dot(self.sn), v.dot(self.tn), v.dot(self.nn))

	def l2w(self, v: 'geo.Vector') -> 'geo.Vector':
		"""
		l2w()

		Transform a `geo.Vector` in the local system
		to the world system
		"""
		return geo.Vector(self.sn.x * v.x + self.tn.x * v.x + self.nn.x * v.x,
		                  self.sn.y * v.y + self.tn.y * v.y + self.nn.y * v.y,
		                  self.sn.z * v.z + self.tn.z * v.z + self.nn.z * v.z)

	def f(self, wo_w: 'geo.Vector', wi_w: 'geo.Vector', flags: 'BDFType' = BDFType.ALL) -> 'Spectrum':
		wi = self.w2l(wi_w)
		wo = self.w2l(wo_w)

		if (wi_w.dot(self.ng)) * (wo_w.dot(self.ng)) < 0.:
			# no transmission
			flags = BDFType(flags & ~BDFType.TRANSMISSION)
		else:
			# no reflection
			flags = BDFType(flags & ~BDFType.REFLECTION)

		f = Spectrum(0.)

		for bdf in self.bdfs:
			if bdf.match_flag(flags):
				f += bdf.f(wo, wi)

		return f

	def rho_hd(self, wo: 'geo.Vector', flags: 'BDFType' = BDFType.ALL, sqrt_samples: INT = 6,
	           rng=np.random.rand) -> 'Spectrum':
		"""
		Computs hemispherical-directional reflectance function.

		"""
		from pytracer.sampler import stratified_sample_2d
		nSamples = sqrt_samples * sqrt_samples
		smp = stratified_sample_2d(sqrt_samples, sqrt_samples, rng=rng)

		sp = Spectrum(0.)
		for bdf in self.bdfs:
			if bdf.match_flag(flags):
				sp += bdf.rho_hd(wo, smp)

		return sp

	def rho_hh(self, flags: 'BDFType' = BDFType.ALL, sqrt_samples: INT = 6, rng=np.random.rand) -> 'Spectrum':
		"""
		Computs hemispherical-hemispherical reflectance function.

		"""
		from pytracer.sampler import stratified_sample_2d
		nSamples = sqrt_samples * sqrt_samples
		smp_1 = stratified_sample_2d(sqrt_samples, sqrt_samples, rng=rng)
		smp_2 = stratified_sample_2d(sqrt_samples, sqrt_samples, rng=rng)

		sp = Spectrum(0.)
		for bdf in self.bdfs:
			if bdf.match_flag(flags):
				sp += bdf.rho_hh(nSamples, smp_1, smp_2)

		return sp



