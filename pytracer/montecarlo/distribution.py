"""
distributino.py

pytracer.montecarlo package

Models empirical distributions.

Created by Jiayao on Aug 8, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer import *

__all__ = ['Distribution1D', 'Distribution2D']


# Utility Classes
class Distribution1D(object):
	"""
	Distribution1D Class

	Piecewise-constant 1D function's
	PDF and CDF and performs sampling.
	"""

	def __init__(self, arr: [FLOAT]):
		"""
		Noted len(arr) + 1 is needed
		for CDF.
		"""
		self.cnt = len(arr)
		self.func = arr.copy()
		self.cdf = np.empty(self.cnt + 1, dtype=FLOAT)

		# compute integral of step function at $x_i$
		self.cdf[0] = 0.
		for i in range(1, self.cnt + 1):
			self.cdf[i] = self.cdf[i - 1] + arr[i - 1] / self.cnt

		# transform step function into CDF
		self.cdf_raw = self.cdf[-1]
		self.cdf /= self.cdf_raw

	def __repr__(self):
		return "{}\nSteps: {}\n".format(self.__class__, self.cnt)

	def sample_cont(self, u: FLOAT) -> [FLOAT, FLOAT]:
		"""
		sample_cont()

		Use given random sample `u` to
		sample from its distribution.
		Returns the r.v. value and sampled
		pdf: [rv, pdf].
		"""
		# surrounding CDF segment
		idx = np.searchsorted(self.cdf, u) - 1
		off = max(0, idx)

		# compute offset
		du = (u - self.cdf[off]) / (self.cdf[off + 1] - self.cdf[off])

		# compute pdf for sampled offset
		# and r.v. value
		return [(du + off) / self.cnt, self.func[off] / self.cdf_raw]

	def sample_dis(self, u: FLOAT) -> [FLOAT, FLOAT]:
		"""
		sample_dis()

		Use given random sample `u` to
		sample from its distribution.
		Returns the r.v. value and sampled
		pdf: [rv, pdf].
		"""
		# surrounding CDF segment
		idx = np.searchsorted(self.cdf, u) - 1
		off = max(0, idx)

		return [off, self.func[off] / (self.cdf_raw * self.cnt)]


class Distribution2D(object):
	"""
	Distribution2D Class

	Piecewise-constant 2D function's
	PDF and CDF and performs sampling.
	"""

	def __init__(self, arr: 'np.ndarray'):
		"""
		arr is the sample values, with shape being
		n_v * n_u
		"""
		nv, nu = np.shape(arr)
		self.p_cond_v = []

		# conditional distribution p(u|v)
		for i in range(nv):
			self.p_cond_v.append(Distribution1D(arr[i, :]))

		# marginal distribution p(v)
		marginal_v = []
		for i in range(nv):
			marginal_v.append(self.p_cond_v[i].cdf_raw)
		self.p_marg_v = Distribution1D(marginal_v)

	def __repr__(self):
		return "{}\nSteps: {}\n".format(self.__class__, self.cnt)

	def sample_cont(self, u0: FLOAT, u1: FLOAT) -> [list, FLOAT]:
		"""
		sample_cont()

		Use given random samples `u0`
		and `u1` to sample from distribution.

		First sampling from p(v) then from p(u|v).
		Returns [[u, v], pdf]
		"""
		v, pdf1 = self.p_marg_v.sample_cont(u1)
		v = np.clip(util.ftoi(v * self.p_marg_v.cnt), 0, self.p_marg_v.cnt - 1).astype(INT)

		u, pdf0 = self.p_cond_v[v].sample_cont(u0)

		return [[u, v], pdf0 * pdf1]

	def pdf(self, u: FLOAT, v: FLOAT) -> FLOAT:
		"""
		pdf()

		Value of the pdf given a sample value
		"""
		ui = np.clip(util.ftoi(u * self.p_cond_v[0].cnt), 0, self.p_cond_v[0].cnt - 1).astype(INT)
		vi = np.clip(util.ftoi(v * self.p_marg_v.cnt), 0, self.p_marg_v.cnt - 1).astype(INT)

		if self.p_cond_v[vi].cdf_raw * self.p_marg_v.cdf_raw == 0.:
			return 0.

		return (self.p_cond_v[vi].func[ui] * self.p_marg_v.func[vi]) / \
		       (self.p_cond_v[vi].cdf_raw * self.p_marg_v.cdf_raw)




