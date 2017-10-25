"""
head_mesh.py

Head mesh constructed using
triangle mesh.
"""
from pytracer import *
from pytracer.shape import (Sphere, create_triangle_mesh)
from pytracer.geometry import (Vector, Point)
from pytracer.transform import (Transform, AnimatedTransform)
from pytracer.aggregate import (BVH, GeometricPrimitive)
from pytracer.texture import ConstantTexture
from pytracer.material import (MatteMaterial, UberMaterial)
from pytracer.light import (SpotLight, DiffuseAreaLight, InfiniteAreaLight)
from pytracer.scene import Scene
from pytracer.filter import LanczosSincFilter
from pytracer.film import ImageFilm
from pytracer.camera import PerspectiveCamera
from examples.head_model import model as head_model

__all__ = ['scene', 'configure_camera']

Spectrum.init()  # if use measured material

# Scene
## Lights
main_trans = Transform.translate(Vector(0., 0., -60))
main_shape = Sphere(main_trans, main_trans.inverse(), False, 3, -3, 3, 360.)
main_light = DiffuseAreaLight(main_trans, Spectrum(10.), 2, main_shape)

lights =[main_light]

## Backdrop
back_param = {
	'indices': [0, 1, 2, 2, 0, 3],
	'P': [-20, 0, -10,
	   20, 0, -10,
	   20, 30, -10,
	   -20, 30, -10]
}
back_Kd = ConstantTexture(Spectrum([.1, .1, .1]))
back_sigma = ConstantTexture(0.)
back_mat = MatteMaterial(back_Kd, back_sigma)
back_trans = Transform.translate(Vector(0., -5., 0.))
back_shape = create_triangle_mesh(back_trans, back_trans.inverse(), False, back_param)
back = GeometricPrimitive(back_shape, back_mat)


## Head mesh
Kd = ConstantTexture(Spectrum([.25, .25, .25]))
Ks = ConstantTexture(Spectrum([.25, .25, .25]))
Kr = ConstantTexture(Spectrum([0., 0., 0.]))
Kt = ConstantTexture(Spectrum([.25, .25, .25]))
roughness = ConstantTexture(0.1)
opacity = ConstantTexture(Spectrum([1., 1., 1.]))
eta = ConstantTexture(1.5)
head_mat = UberMaterial(Kd, Ks, Kr, Kt, roughness, opacity, eta)

head_trans = Transform.translate(Vector(-3, 6., -15))
head_shape = create_triangle_mesh(head_trans, head_trans.inverse(), False, head_model)
head = GeometricPrimitive(head_shape, head_mat)

## Aggregation and Scene
aggs = BVH([head, back])
scene = Scene(aggs, lights, None)


# Camera
def configure_camera(x_res=50, y_res=50, file_name='tmp.png'):
	fil = LanczosSincFilter(4., 4., 3.)
	film = ImageFilm(x_res, y_res, fil, [0., 1., 0., 1.], file_name)
	c_trans = Transform.translate(Vector(-13.,-1,-35))
	cam_trans = AnimatedTransform(c_trans, 0., c_trans, 0.)
	camera = PerspectiveCamera(cam_trans, [0., 1., 0., 1.], 0., 0., 0., 1e100, 90., film)
	return camera



