"""
parameter.py

pytracer.interface package

Container for parameters and instance
creation.


Created by Jiayao on Aug 21, 2017
"""
from __future__ import absolute_import

__all__ = ['Param']


class Param(object):
	"""Parameter Class"""
	PARAM_TYPE = ['float', 'int', 'bool', 'vector', 'normal', 'spectrum', 'point',
	              'string']

	def __init__(self):
		self.data = {}
		self._reset_param()

	def __repr__(self):
		return "{}\n{}\n".format(self.__class__, self.data)

	def _reset_param(self):
		"""Reset parameters"""
		for key in Param.PARAM_TYPE:
			if key not in self.data:
				self.data[key] = []
			else:
				self.data[key] = []

	def push_back(self, tp: 'str', data):
		"""Data will be pushed as a whole (i.e., append)."""
		if tp not in Param.PARAM_TYPE:
			raise KeyError

		self.data[tp].append(data)

	def enum(self, tp: 'str'):
		"""Returns the next data of type tp."""
		if tp not in Param.PARAM_TYPE:
			raise KeyError
		elif len(self.data[tp]) == 0:
			return None
		else:
			return self.data[tp].pop(0)

	def fetch(self, tp: 'str'):
		"""Returns the list of data of type tp."""
		if tp not in Param.PARAM_TYPE:
			raise KeyError
		else:
			return self.data[tp]

	def has_more(self, tp: 'str'):
		if tp not in Param.PARAM_TYPE:
			raise KeyError
		else:
			return len(self.data[tp]) > 0

