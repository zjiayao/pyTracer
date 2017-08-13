"""
material.py

Model materials

Created by Jiayao on Aug 3, 2017
"""
from src.volume.volume import *

class Material(object, metaclass=ABCMeta):
	"""
	Material Class
	"""
	def __repr__(self):
		return "{}\n".format(self.__class__)

	@abstractmethod	
	def get_BSDF(self, dg_g: 'DifferentialGeometry', dg_s: 'DifferentialGeometry') -> 'BSDF':
		"""
		dg_g: Geometric DG object
		dg_s: Shading DG object
		"""
		raise NotImplementedError('src.core.material.{}.get_BSDF: abstract method '
									'called'.format(self.__class__)) 	

	def get_BSSRDF(self, dg_g: 'DifferentialGeometry', dg_s: 'DifferentialGeometry') -> 'BSSRDF':
		"""
		get_BSSRDF()

		For translucent materials. 
		dg_g: Geometric DG object
		dg_s: Shading DG object
		"""
		return None
	@staticmethod
	@jit
	def bump(d: 'Texture', dgg: 'DifferentialGeometry', dgs: 'DifferentialGeometry'):
		"""
		bump()

		Compute bump mapping from `Texture` d,
		dgg accounts for geometry and dgs
		for shading
		"""
		# offset position, find texture
		dg = dgs.copy()
		## shift du in u
		du = .5 * (np.fabs(dgs.dudx) + np.fabs(dgs.dudy))
		if du == 0.:
			du = 0.01
		dg.p = dgs.p + du * dgs.dpdu
		dg.u = dgs.u + du
		dg.nn = normalize(Normal.fromVector(dgs.dpdu.cross(dgs.dpdv) + du * dgs.dndu))

		u_disp = d(dg)

		# shift du in v
		dv = .5 * (np.fabs(dgs.dvdx) + np.fabs(dgs.dvdy))
		if dv == 0.1:
			dv = 0.01
		dg.p = dgs.p + dv * dgs.dpdv
		dg.u = dgs.u
		dg.v = dgs.v + dv
		dg.nn = normalize(Normal.fromVector(dgs.dpdu.cross(dgs.dpdv) + dv * dgs.dndv))

		v_disp = d(dg)

		disp = d(dgs)

		# compute bump mapped dg
		dg = dgs.copy()

		# Vector * Normal -> Vector
		dg.dpdu = dgs.dpdu + (u_disp - disp) / du * dgs.nn + disp * Vector.fromNormal(dgs.dndu)
		dg.dpdv = dgs.dpdv + (v_disp - disp) / dv * dgs.nn + disp * Vector.fromNormal(dgs.dndv)
		dg.nn = Normal.fromVector(normalize(dg.dpdu.cross(dg.dpdv)))
		if (dgs.shape.ro ^ dgs.shape.transform_swaps_handedness):
			dg.nn *= -1.
 
		# orient shding normal to match goemtric normal
		dg.nn = face_forward(dg.nn, dgg.nn)

		return dg


			
class MatteMaterial(Material):
	"""
	MatteMaterial Class

	Models prurely diffuse surfaces.

	Kd: Specular Diffuse Reflection Value
	sigma: Roughness. Returns Lambertian if
			0 at some point; OrenNayer otherwise

	"""
	def __init__(self, Kd: 'Texture', sigma: 'Texture', bump_map: 'Texture'):
		"""
		Kd: `Spectrum` `Texture`
		sigma, bump_map: `FLOAT` `Texture`
		"""
		self.Kd = Kd
		self.sigma = sigma
		self.bump_map = bump_map


	def get_BSDF(self, dg_g: 'DifferentialGeometry', dg_s: 'DifferentialGeometry') -> 'BSDF':
		# possibly bump mapping
		if self.bump_map is not None:
			dgs = self.bump(self.bump_map, dg_g, dg_s)
		else:
			dgs = dg_s.copy()

		bsdf = BSDF(dgs, dg_g.nn)

		# evalutae textures
		r = self.Kd(dgs).clip()
		sigma = np.clip(self.sigma(dgs), 0., 90.)
		if sigma == 0.:
			# use Lambertian
			bsdf.push_back(Lambertian(r))
		else:
			# use Oren-Nayer
			bsdf.push_back(OrenNayar(r, sigma))
		return bsdf


class PlasticMaterial(Material):
	"""
	PlasticMaterial Class

	Models plastic as a mixture
	of diffuse and glossy scatterings.

	Kd: Specular Diffuse Reflection Value
	Ks: Specular Glossy Reflection Value
	rough: Roughness, determines specular highlight

	"""
	def __init__(self, Kd: 'Texture', Ks: 'Texture', roughness: 'Texture', bump_map: 'Texture'):
		"""
		Kd: `Spectrum` `Texture`
		sigma, bump_map: `FLOAT` `Texture`
		"""
		self.Kd = Kd
		self.Ks = Ks
		self.roughness = roughness
		self.bump_map = bump_map


	def get_BSDF(self, dg_g: 'DifferentialGeometry', dg_s: 'DifferentialGeometry') -> 'BSDF':
		# possibly bump mapping
		if self.bump_map is not None:
			dgs = Material.bump(self.bump_map, dg_g, dg_s)
		else:
			dgs = dg_s.copy()

		bsdf = BSDF(dgs, dg_g.nn)

		# diffuse
		kd = self.Kd(dgs).clip()
		diff = Lambertian(kd)
		fresnel = FresnelDielectric(1.5, 1.)

		# glossy specular
		ks = self.Ks(dgs).clip()
		rough = self.roughness(dgs)
		spec = Microfacet(ks, fresnel, Blinn(1. / rough))

		bsdf.push_back(diff)
		bsdf.push_back(spec)

		return bsdf



class MixMaterial(Material):
	"""
	MixMaterial Class

	Models mixed materials. Use texture
	spectrum to blend.
	"""
	def __init__(self, m1: 'Texture', m2: 'Texture', scale: 'Texture'):
		"""
		m1, m2: `Material` `Spectrum`
		scale: `Spectrum` `Texture` for blend
		"""
		self.m1 = m1
		self.m2 = m2
		self.scale = scale


	def get_BSDF(self, dg_g: 'DifferentialGeometry', dg_s: 'DifferentialGeometry') -> 'BSDF':
		"""
		BSDF in m1 (i.e., `m1.bdfs`) is modified
		"""
		b1 = self.m1.get_BSDF(dg_g, dg_s)
		b2 = self.m2.get_BSDF(dg_g, dg_s)

		s1 = self.scale(dg_s).clip()
		s2 = (Spectrum(1.) - s1).clip()

		for i, b in b1.bdfs:
			b1.bdfs[i] = ScaledBDF(b, s1)
		for i, b in b2.bdfs:
			b1.push_back(ScaledBDF(b, s2))

		return b1


class MeasuredMaterial(Material):
	"""
	MeasuredMaterial Class

	Models materials using measured BSDF,
	either irregular or regular.
	"""
	def __init__(self, filename: 'str', bump_map: 'Texture'):
		"""
		filename: filename of measured material
		bump_map: `FLOAT` `Texture`

		Irregular Sampled Isotropic BRDF:
			*.brdt
		Regular Halfangle BRDF:
			*.merl (Binary)
		"""
		self.bump_map = bump_map
		self.regular_data = None
		self.theta_phi_tree = None
		self.theta_phi_data = None

		dire, file = os.path.split(filename)
		name, ext = os.path.splitext(file)

		# Irregular Sampled Isotropic BRDF
		if ext == '.brdf' or ext == '.BRDF':
			if file in IrIsotropicData:
				# already read
				self.theta_phi_tree, self.theta_phi_sample = IrIsotropicData[file]
				return

			# read file
			if not os.path.exists(filename):
				raise RuntimeError('src.core.material.{}.__init__() '
						': Cannot load {}, file not found'.format(self.__class__,
								file))

			val = []
			with open(filename, 'r') as f:
				for line in f:
					line = line.split()
					if not len(line) or line[0] == '#':
						# skip comments
						continue
					val.extend(np.array(line).astype(FLOAT))
			val = np.array(val,dtype=FLOAT)
			len_val = len(val)

			# process data
			num_wls = INT(val[0])
			if (len_val - 1 - num_wls) % (4 + num_wls) != 0:
				raise RuntimeError('src.core.material.{}.__init__() '
						': Cannot load {}, format error'.format(self.__class__,
								file))
			num_smp = INT((len_val - 1 - num_wls) / (4 + num_wls))
			wls = val[1:1+num_wls]

			samples = []
			tree_data = np.empty([num_smp, 4])

			pos = 1 + num_wls
			cnt = 0
			while pos < len_val:
				thi, phi, tho, pho = tree_data[cnt,:] = val[pos:pos+4]
				cnt += 1
				pos += 4
				wo = spherical_direction(np.sin(tho), np.cos(tho), pho)
				wi = spherical_direction(np.sin(thi), np.cos(thi), phi)
				s = Spectrum.fromSampled(wls, val[pos:pos+num_wls], num_wls)
				p = BRDF_remap(wo, wi)
				samples.append(IrIsotropicBRDFSample(p, s))
				pos += num_wls

			val = [] # release memory

			self.theta_phi_samples = np.array(samples, dtype=object)
			self.theta_phi_tree = KdTree(tree_data)
			IrIsotropicData[file] = [self.theta_phi_samples, self.theta_phi_tree]
		

		elif ext == '.binary':
			# Regular Halfangle BRDF
			self.n_theta_h = 90
			self.n_theta_d = 90
			self.n_phi_d = 180
			length = self.n_theta_h * self.n_theta_d * self.n_phi_d

			#with open(filename, 'rb') as f:
			raise NotImplementedError('src.core.material.{}.__init__() '
					': Has not support MERL yet'.format(self.__class__,
							file, ext))




		else:
			raise IOError('src.core.material.{}.__init__() '
					': Cannot load {}, unknown type {}'.format(self.__class__,
							file, ext))


	def get_BSDF(self, dg_g: 'DifferentialGeometry', dg_s: 'DifferentialGeometry') -> 'BSDF':
		"""
		BSDF in m1 (i.e., `m1.bdfs`) is modified
		"""
		if self.bump_map is not None:
			dgs = Material.bump(self.bump_map, dg_g, dg_s)
		else:
			dgs = dg_s.copy()

		bsdf = BSDF(dgs, dg_g.nn)			

		if self.regular_data is not None:
			# regular tabulate
			bsdf.push_back(ReHalfangleBRDF(self.regular_data), self.n_theta_h, self.n_theta_d,
						self.n_phi_d)

		elif self.theta_phi_tree is not None:
			# irregular
			bsdf.push_back(IrIsotropicBRDF(self.theta_phi_tree, self.theta_phi_samples))

		return bsdf



class SubsurfaceMaterial(Material):
	"""
	SubsurfaceMaterial Class

	Models translucent objects
	"""
	def __init__(self, scale: FLOAT, kr: 'Texture', sigma_a: 'Texture',
					sigma_prime_s: 'Texture', eta: 'Texture', bump_map: 'Texture'):
		self.scale = scale
		self.kr = kr
		self.sigma_a = sigma_a
		self.sigma_prime_s = sigma_prime_s
		self.eta = eta
		self.bump_map = bump_map

	def get_BSDF(self, dg_g: 'DifferentialGeometry', dg_s: 'DifferentialGeometry') -> 'BSDF':
		if self.bump_map is not None:
			dgs = Material.bump(self.bump_map, dg_g, dg_s)
		else:
			dgs = dg_s.copy()
		R = self.kr(dgs).clip()
		bsdf = BSDF(dgs, dg_g.nn)

		if not R.is_black():
			bsdf.push_back(SpecularReflection(R, FresnelDielectric(1., self.eta(dgs))))

		return bsdf


	def get_BSSRDF(self, dg_g: 'DifferentialGeometry', dg_s: 'DifferentialGeometry') -> 'BSSRDF':
		"""
		get_BSSRDF()

		Specified in terms of reciprocal
		distance ($m^{-1}$).
		"""
		return BSSRDF(self.scale * self.sigma_a(dg_s), self.scale * self.sigma_prime_s(dg_s), self.eta(dg_s))




class KdSubsurfaceMaterial(Material):
	"""
	SubsurfaceMaterial Class

	Models translucent objects
	"""
	def __init__(self, kd: 'Texture', kr: 'Texture', mfp: 'Texture',
					eta: 'Texture', bump_map: 'Texture'):
		self.kd = kd
		self.kr = kr
		self.mfp = mfp # mean free path
		self.eta = eta
		self.bump_map = bump_map

	def get_BSDF(self, dg_g: 'DifferentialGeometry', dg_s: 'DifferentialGeometry') -> 'BSDF':
		if self.bump_map is not None:
			dgs = Material.bump(self.bump_map, dg_g, dg_s)
		else:
			dgs = dg_s.copy()

		R = self.kr(dgs).clip()
		bsdf = BSDF(dgs, dg_g.nn)

		if not R.is_black():
			bsdf.push_back(SpecularReflection(R, FresnelDielectric(1., self.eta(dgs))))

		return bsdf


	def get_BSSRDF(self, dg_g: 'DifferentialGeometry', dg_s: 'DifferentialGeometry') -> 'BSSRDF':
		"""
		get_BSSRDF()

		Specified in terms of reciprocal
		distance ($m^{-1}$).
		"""
		e = self.eta(dg_s)
		sigma_a, sigma_prime_s = subsurface_from_diffuse(self.kd(dg_s).clip(), self.mfp(dg_s), e)

		return BSSRDF(sigma_a, sigma_prime_s, e)

