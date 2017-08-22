"""
api.py

pytracer.interface package

Interfacing from the scene descriptions.


Created by Jiayao on Aug 21, 2017
"""
from __future__ import absolute_import
from typing import TYPE_CHECKING
import pytracer.utility as util
if TYPE_CHECKING:
	from pytracer.interface.option import Option
	from pytracer.interface.parameter import Param
	from pytracer.transform import Transform


def system_init(option: Option):
	from pytracer.spectral import Spectrum
	import pytracer.interface as inter

	if inter.API_STATUS != inter.API_UNINIT:
		raise RuntimeError("system_init() already called.")

	inter.API_STATUS = inter.API_OPTIONS

	inter.GLOBAL_OPTION = option

	inter.RENDER_OPTION = inter.RenderOption()

	inter.GRAPHICS_STATE = GraphicsState()

	Spectrum.init()


def system_clean():
	import pytracer.interface as inter
	if inter.API_STATUS == inter.API_UNINIT:
		return

	inter.API_STATUS = inter.API_UNINIT

	inter.GLOBAL_OPTION = None

	inter.RENDER_OPTION = None


def check_system_inited(api_func):
	import pytracer.interface as inter
	if inter.API_STATUS == inter.API_UNINIT:
		raise RuntimeError("{}: system not inited.".format(api_func.__name__))
	return api_func


@check_system_inited
def trans_identity():
	import pytracer.interface as inter
	import pytracer.transform as trans
	for i, _ in enumerate(inter.TRANSFORM_SET):
		inter.TRANSFORM_SET[i] = trans.Transform()


@check_system_inited
def trans_translate(dx):
	import pytracer.interface as inter
	import pytracer.geometry as geo
	import pytracer.transform as trans
	for i, _ in enumerate(inter.TRANSFORM_SET):
		inter.TRANSFORM_SET[i] *= trans.Transform.translate(geo.Vector.from_arr(dx))


@check_system_inited
def trans_rotate(angle, dx: list):
	import pytracer.interface as inter
	import pytracer.geometry as geo
	import pytracer.transform as trans
	for i, _ in enumerate(inter.TRANSFORM_SET):
		inter.TRANSFORM_SET[i] *= trans.Transform.rotate(angle, geo.Vector.from_arr(dx))


@check_system_inited
def trans_scale(sc: list):
	import pytracer.interface as inter
	import pytracer.transform as trans
	for i, _ in enumerate(inter.TRANSFORM_SET):
		inter.TRANSFORM_SET[i] *= trans.Transform.scale(sc[0], sc[1], sc[2])


@check_system_inited
def trans_look_at(eye: list, at: list, up: list):
	import pytracer.interface as inter
	import pytracer.geometry as geo
	import pytracer.transform as trans
	for i, _ in enumerate(inter.TRANSFORM_SET):
		inter.TRANSFORM_SET[i] *= trans.Transform.look_at(geo.Point.from_arr(eye),
		                                                  geo.Point.from_arr(at),
		                                                  geo.Vector.from_arr(up))


@check_system_inited
def trans_concat(trans: Transform):
	import pytracer.interface as inter
	for i, _ in enumerate(inter.TRANSFORM_SET):
		inter.TRANSFORM_SET[i] *= trans


@check_system_inited
def trans_set(trans: Transform):
	import pytracer.interface as inter
	for i, _ in enumerate(inter.TRANSFORM_SET):
		inter.TRANSFORM_SET[i] = trans


# coordinate system
@check_system_inited
def add_coordinate_system(name: str):
	"""Make a named copy of the current transformations."""
	import pytracer.interface as inter
	inter.COORDINATE_SYSTEM[name] = inter.TRANSFORM_SET


@check_system_inited
def set_coordinate_system(name: str):
	"""Make a named copy of the current transformations."""
	import pytracer.interface as inter
	if name not in inter.COORDINATE_SYSTEM:
		raise RuntimeError

	inter.TRANSFORM_SET = inter.COORDINATE_SYSTEM[name]


# Render Options
@check_system_inited
def set_transformation_time(start=0., end=1.):
	import pytracer.interface as inter
	inter.RENDER_OPTION.trans_start_time = start
	inter.RENDER_OPTION.trans_end_time = end


# Filters
@check_system_inited
def set_pixel_filter(name: str, param: Param):
	import pytracer.interface as inter
	inter.RENDER_OPTION.filter_name = name.lower()
	inter.RENDER_OPTION.filter_param = param


@check_system_inited
def set_film(name: str, param: Param):
	import pytracer.interface as inter
	inter.RENDER_OPTION.film_name = name.lower()
	inter.RENDER_OPTION.film_param = param
	
	
@check_system_inited
def set_camera(name: str, param: Param):
	import pytracer.interface as inter
	inter.RENDER_OPTION.camera_name = name.lower()
	inter.RENDER_OPTION.camera_param = param
	inter.RENDER_OPTION.cam2wld = [trans.inverse() for trans in inter.TRANSFORM_SET]
	inter.COORDINATE_SYSTEM["camera"] = inter.RENDER_OPTION.cam2wld


@check_system_inited
def set_sampler(name: str, param: Param):
	import pytracer.interface as inter
	inter.RENDER_OPTION.sampler_name = name.lower()
	inter.RENDER_OPTION.sampler_param = param


@check_system_inited
def set_aggregator(name: str, param: Param):
	import pytracer.interface as inter
	inter.RENDER_OPTION.aggregator_name = name.lower()
	inter.RENDER_OPTION.aggregator_param = param


@check_system_inited
def set_renderer(name: str, param: Param):
	import pytracer.interface as inter
	inter.RENDER_OPTION.renderer_name = name.lower()
	inter.RENDER_OPTION.renderer_param = param


@check_system_inited
def set_surface(name: str, param: Param):
	import pytracer.interface as inter
	inter.RENDER_OPTION.surface_name = name.lower()
	inter.RENDER_OPTION.surface_param = param


@check_system_inited
def set_volume(name: str, param: Param):
	import pytracer.interface as inter
	inter.RENDER_OPTION.volume_name = name.lower()
	inter.RENDER_OPTION.volume_param = param


# Scene Description
@check_system_inited
def world_begin():
	import pytracer.interface as inter
	inter.API_STATUS = inter.API_WORLD
	trans_identity()
	add_coordinate_system("world")


@check_system_inited
def attribute_begin():
	import pytracer.interface as inter
	inter.GRAPHICS_STATE_STACK.append(inter.GRAPHICS_STATE)
	inter.TRANSFORM_STACK.append(inter.TRANSFORM_SET)


@check_system_inited
def attribute_end():
	import pytracer.interface as inter
	if len(inter.GRAPHICS_STATE_STACK) == 0:
		util.logging('Error', 'Unmatched attribute end, ignoring')
		return

	inter.GRAPHICS_STATE = inter.GRAPHICS_STATE_STACK.pop()
	inter.TRANSFORM_SET = inter.TRANSFORM_STACK.pop()


@check_system_inited
def transform_begin():
	import pytracer.interface as inter
	inter.TRANSFORM_STACK.append(inter.TRANSFORM_SET)


@check_system_inited
def transform_end():
	import pytracer.interface as inter
	if len(inter.TRANSFORM_STACK) == 0:
		util.logging('Error', 'Unmatched attribute end, ignoring')
		return

	inter.TRANSFORM_SET = inter.TRANSFORM_STACK.pop()




# Local Classes
class GraphicsState(object):
	"""Holds the graphics states."""
	def __init__(self):
		pass

	def __repr__(self):
		return "{}\n".format(self.__class__)





