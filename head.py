from pytracer import *
from pytracer.shape import (Sphere, Disk, create_triangle_mesh, create_loop_subdiv)
from pytracer.geometry import (Vector, Point)
from pytracer.transform import (Transform, AnimatedTransform)
from pytracer.aggregate import (BVH, SimpleAggregate, GeometricPrimitive, GridAccel)
from pytracer.texture import ConstantTexture
from pytracer.material import (MatteMaterial, UberMaterial, MeasuredMaterial, MirrorMaterial, MetalMaterial, GlassMaterial)
from pytracer.light import (SpotLight, DiffuseAreaLight, InfiniteAreaLight)
from pytracer.scene import Scene
from pytracer.sampler import StratifiedSampler
from pytracer.filter import (BoxFilter, LanczosSincFilter)
from pytracer.film import ImageFilm
from pytracer.camera import (PinholeCamera, OrthoCamera, PerspectiveCamera)
from pytracer.integrator import (PathIntegrator, DirectLightingIntegrator, WhittedIntegrator)
from pytracer.renderer import SamplerRenderer
from examples.head_model import model as head_model
Spectrum.init()
print("Dependencies loaded")

x_res = 10
y_res = 10
spp_x = 1
spp_y = 1
file = 'test_7.png'

np.random.seed(1)

# camera
fil = LanczosSincFilter(4., 4., 3.)  #  BoxFilter(.5, .5)
film = ImageFilm(x_res, y_res, fil, [0., 1., 0., 1.], file)
film = ImageFilm(x_res, y_res, fil, [0., 1., 0., 1.], file)
c_trans = Transform.translate(Vector(-16.,-8.,-45))#Transform.look_at(Point(5., 5., 100.), Point(0., 0., -1), Vector(0., 1., 0.))
cam_trans = AnimatedTransform(c_trans, 0., c_trans, 0.)
camera = PerspectiveCamera(cam_trans, [0., 1., 0., 1.], 0., 0., 0., 1e100, 90., film)
print("Camera assembled")

# sampler
sampler = StratifiedSampler(0, x_res, 0, y_res, spp_x, spp_y, True, 0., 0.)
print("Sampler built")

# surface integrator
surf_int = DirectLightingIntegrator()
print("Integrator initialized")

# lights
main_trans = Transform.translate(Vector(0., 0., -100))# Transform.rotate(180., Vector(0., 1., 0.)) * Transform.translate(Vector(0., 15., 0))
main_shape = Disk(main_trans, main_trans.inverse(),False, 0., 1.5, 0., 360.)
ls = Sphere(main_trans, main_trans.inverse(), False, 3, -3, 3, 360.)
main_light = DiffuseAreaLight(main_trans, Spectrum(10.), 2, ls)

lights =[main_light]
print("Lights built")

# shape
# Backdrop
back_param = {
	'indices': [0, 1, 2, 2, 0, 3],
	'P': [-10, 0, -10, 10, 0, -10, 10, 9, -10, -10, 9, -10]
}
back_Kd = ConstantTexture(Spectrum([.1, .1, .1]))
back_sigma = ConstantTexture(0.)
back_mat = MatteMaterial(back_Kd, back_sigma)
back_trans = Transform.translate(Vector(0., -5., 0.))
back_shape = create_triangle_mesh(back_trans, back_trans.inverse(), False, back_param)
back = GeometricPrimitive(back_shape, back_mat)

# tmp = create_loop_subdiv(back_trans, back_trans.inverse(),False, back_param)

Kd = ConstantTexture(Spectrum([.25, .25, .25]))
Ks = ConstantTexture(Spectrum([.25, .25, .25]))
Kr = ConstantTexture(Spectrum([0., 0., 0.]))
Kt = ConstantTexture(Spectrum([.25, .25, .25]))
roughness = ConstantTexture(0.1)
opacity = ConstantTexture(Spectrum([1., 1., 1.]))
eta = ConstantTexture(1.5)
head_trans = Transform.translate(Vector(0-3, 6., -15))#Transform.scale(.1, .1, .1) * Transform.rotate(180, Vector(0., 1., 0.)) * Transform.translate(Vector(0., -.2, 0.))
head_mat = UberMaterial(Kd, Ks, Kr, Kt, roughness, opacity, eta)
head_shape = Sphere(head_trans, head_trans.inverse(),False, 1.5, -1.5, 1.5, 360)
# head_shape = create_triangle_mesh(head_trans, head_trans.inverse(), False, head_model)
head_shape = create_loop_subdiv(head_trans, head_trans.inverse(), False, head_model)
head = GeometricPrimitive(head_shape, head_mat)




# head2_trans = Transform.translate(Vector(1.5, 1.5, -10))#Transform.scale(.1, .1, .1) * Transform.rotate(180, Vector(0., 1., 0.)) * Transform.translate(Vector(0., -.2, 0.))
# lam_arr = []
# v_arr = []
# with open('./scene/texture/metal/Cu.eta.spd', 'r') as f:
# 	for line in f:
# 		lam, v = line.split()
# 		lam_arr.append(FLOAT(lam))
# 		v_arr.append(FLOAT(v))
# eta = ConstantTexture(Spectrum.from_sampled(lam_arr, v_arr))
# lam_arr = []
# v_arr = []
# with open('./scene/texture/metal/Cu.k.spd', 'r') as f:
# 	for line in f:
# 		lam, v = line.split()
# 		lam_arr.append(FLOAT(lam))
# 		v_arr.append(FLOAT(v))
# k = ConstantTexture(Spectrum.from_sampled(lam_arr, v_arr))
# head2_mat = GlassMaterial(ConstantTexture(Spectrum(.8)), ConstantTexture(Spectrum(.7)), ConstantTexture(1.5))
# head2_shape = Sphere(head2_trans, pythonhead2_trans.inverse(),
#                     False, .8, -.8, .8, 360)
# head2 = GeometricPrimitive(head2_shape, head2_mat)
# head = GeometricPrimitive(head_shape, head2_mat)

# agg
# aggs = SimpleAggregate([back, head], True)
aggs = BVH([head])
print("Aggregates built using {}\n{}".format(aggs.split_method, aggs.world_bound()))

# scene
scene = Scene(aggs, lights, None)
print("Scene constructed")

renderer = SamplerRenderer(sampler, camera, surf_int, None)
print("Renderer built, proceed to render...")

renderer.render(scene)
