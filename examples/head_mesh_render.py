"""
head_mesh_render.py


Construct the scene from model
and render it.
"""
from pytracer import *
from pytracer.sampler import StratifiedSampler
from pytracer.integrator import DirectLightingIntegrator
from pytracer.renderer import SamplerRenderer
import examples.head_map as model

x_res, y_res = 128, 128
x_spr, y_spr = 1, 1

camera = model.configure_camera(x_res=x_res, y_res=y_res, file_name='tmp.png')
sampler = StratifiedSampler(0, x_res, 0, y_res, x_spr, y_spr, True, 0., 0.)
surf_int = DirectLightingIntegrator()
renderer = SamplerRenderer(sampler, camera, surf_int, None)
image = renderer.render(model.scene)
