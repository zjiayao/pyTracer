"""
option.py

pytracer.interface package

Parsing options for initialization.


Created by Jiayao on Aug 20, 2017
"""
from __future__ import absolute_import

__all__ = ['Option', 'RenderOption']


class Option(object):
	"""Option Class"""
	def __init__(self, image_file: str="", quick_render=False, n_cores=1):
		self.quick_render = quick_render
		self.image_file = image_file
		self.n_cores = n_cores

	def __repr__(self):
		return "{}\n".format(self.__class__)


class RenderOption(object):
	"""Options for renderer."""
	def __init__(self):
		self.trans_start_time = 0.
		self.trans_end_time = 1.
		self.filter_name = "box"
		self.filter_param = None
		self.film_name = "image"
		self.film_param = None
		self.sampler_name = "stratified"
		self.sampler_param = None
		self.aggregator_name = "simple"
		self.aggregator_param = None
		self.renderer_name = "sampler"
		self.renderer_param = None
		self.surface_name = "directlighting"
		self.surface_param = None
		self.volume_name = ""
		self.volume_param = None
		self.camera_name = "perspective"
		self.camera_param = None
		self.instance = None

		import pytracer.transform as trans
		from pytracer.interface import MAX_TRANSFORM
		self.cam2wld = [trans.Transform()] * MAX_TRANSFORM

	def __repr__(self):
		return "{}".format(self.__class__)