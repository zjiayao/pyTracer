"""
test_render.py


Test by rendering a simple scene.

Created on Oct. 25, 2017 by Jiayao
"""

from __future__ import (absolute_import, print_function, division)


def test_render():
	import numpy as np
	from pytracer import *

	Spectrum.init()

	from pytracer.geometry import (Vector, Point)
	from pytracer.transform import (Transform, AnimatedTransform)

	from pytracer.geometry import (Vector, Point)
	from pytracer.transform import (Transform, AnimatedTransform)

	from pytracer.shape import (Sphere, create_triangle_mesh)
	from pytracer.aggregate import GeometricPrimitive
	from pytracer.light import DiffuseAreaLight
	from pytracer.texture import ConstantTexture
	from pytracer.material import (MatteMaterial, UberMaterial)


	back_param = {
			'indices': [0, 1, 2, 2, 0, 3],
			'P': [-20, 0, -10,
					 20, 0, -10,
					 20, 9, -10,
					 -20, 9, -10]
	}
	back_Kd = ConstantTexture(Spectrum([.1, .1, .1]))
	back_sigma = ConstantTexture(0.)
	back_mat = MatteMaterial(back_Kd, back_sigma)
	back_trans = Transform.translate(Vector(0., -5., 0.))
	back_shape = create_triangle_mesh(back_trans, back_trans.inverse(), False, back_param)
	back = GeometricPrimitive(back_shape, back_mat)


	sphere_transform = Transform.translate(Vector(.5, .5, -6))
	sphere_shape = Sphere(sphere_transform, sphere_transform.inverse(), False,  # whther reverse orientation
											   2., -2., 2., 360.)  # for partial sphere: radius, z_min, z_max, phi_max (in degrees)


	Kd = ConstantTexture(Spectrum([.25, .25, .25]))
	Ks = ConstantTexture(Spectrum([.25, .25, .25]))
	Kr = ConstantTexture(Spectrum([0., 0., 0.]))
	Kt = ConstantTexture(Spectrum([.25, .25, .25]))
	roughness = ConstantTexture(0.1)
	opacity = ConstantTexture(Spectrum([1., 1., 1.]))
	eta = ConstantTexture(1.5)
	sphere_material = UberMaterial(Kd, Ks, Kr, Kt, roughness, opacity, eta)

	sphere = GeometricPrimitive(sphere_shape, sphere_material)
	shapes = [back, sphere]

	light_transform = Transform.translate(Vector(0., 1., 5.))
	light_shape = Sphere(light_transform, light_transform.inverse(), False, 1., -1., 1., 360.)
	light = DiffuseAreaLight(light_transform, Spectrum(10.), 2, light_shape)

	lights = [light]


	from pytracer.scene import Scene
	from pytracer.aggregate import BVH
	aggs = BVH(shapes, False)
	scene = Scene(aggs, lights, None)

	from pytracer.camera import PerspectiveCamera
	from pytracer.film import ImageFilm
	from pytracer.filter import LanczosSincFilter
	from pytracer.sampler import StratifiedSampler


	x_res, y_res = 64, 64


	x_spr, y_spr = 2, 2


	sampler = StratifiedSampler(0, x_res, 0, y_res, x_spr, y_spr, True, 0., 0.)


	trans = Transform.translate(Vector(3.5, 3.5, 1.)) * Transform.look_at(Point(0.1, 0., 0.), Point(0., 0., -1.), Vector(0., 0., 1.))
	cam_trans = AnimatedTransform(trans, 0., trans, 0.)


	film = ImageFilm(xr=x_res, yr=y_res, filt=LanczosSincFilter(4., 4., 3.),
									 crop=[0., 1., 0., 1.], fn='tmp.png')

	camera = PerspectiveCamera(cam_trans, scr_win=[0., 1., 0., 1.], s_open=0.,
														 s_close=0., lensr=0., focald=1e100, fov=90., f=film)

	from pytracer.integrator import DirectLightingIntegrator
	from pytracer.renderer import SamplerRenderer

	surface_integrator = DirectLightingIntegrator()
	renderer = SamplerRenderer(sampler, camera, surface_integrator, None)
	image = renderer.render(scene)

