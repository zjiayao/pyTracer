from __future__ import absolute_import


def test_import():
	import pytracer
	import pytracer.geometry
	import pytracer.geometry.utility
	import pytracer.geometry.diffgeom
	import pytracer.transform
	import pytracer.shape
	import pytracer.shape.disk
	import pytracer.shape.sphere
	import pytracer.shape.cylinder
	import pytracer.shape.triangle
	import pytracer.shape.loopsubdiv
	import pytracer.aggregate
	import pytracer.film
	import pytracer.camera
	import pytracer.filter
	import pytracer.integrator
	import pytracer.light
	import pytracer.material
	import pytracer.montecarlo
	import pytracer.reflection
	import pytracer.renderer
	import pytracer.sampler
	import pytracer.scene
	import pytracer.spectral
	import pytracer.texture
	import pytracer.utility.imageio
	import pytracer.volume
	assert 1