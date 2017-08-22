"""
material.py

Model materials

Created by Jiayao on Aug 3, 2017
"""
from __future__ import absolute_import
import os
from abc import (ABCMeta, abstractmethod)
from pytracer import *
import pytracer.geometry as geo
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from pytracer.texture import Texture
	from pytracer.reflection import BSDF
	from pytracer.volume import BSSRDF


__all__ = ['Material', 'MatteMaterial', 'PlasticMaterial',
           'MixMaterial', 'MeasuredMaterial', 'SubsurfaceMaterial',
           'KdSubsurfaceMaterial', 'UberMaterial']


class Material(object, metaclass=ABCMeta):
	"""
	Material Class
	"""
	def __repr__(self):
		return "{}\n".format(self.__class__)

	@abstractmethod
	def get_bsdf(self, dg_g: 'geo.DifferentialGeometry', dg_s: 'geo.DifferentialGeometry') -> 'BSDF':
		"""
		dg_g: Geometric DG object
		dg_s: Shading DG object
		"""
		raise NotImplementedError('src.core.material.{}.get_bsdf: abstract method '
									'called'.format(self.__class__))

	def get_bssrdf(self, dg_g: 'geo.DifferentialGeometry', dg_s: 'geo.DifferentialGeometry') -> 'BSSRDF':
		"""
		get_bssrdf()

		For translucent materials.
		dg_g: Geometric DG object
		dg_s: Shading DG object
		"""
		return None

	@staticmethod
	def bump(d: 'Texture', dgg: 'geo.DifferentialGeometry', dgs: 'geo.DifferentialGeometry'):
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
		dg.nn = geo.normalize(geo.Normal.from_arr(dgs.dpdu.cross(dgs.dpdv) + du * dgs.dndu))

		u_disp = d(dg)

		# shift du in v
		dv = .5 * (np.fabs(dgs.dvdx) + np.fabs(dgs.dvdy))
		if dv == 0.1:
			dv = 0.01
		dg.p = dgs.p + dv * dgs.dpdv
		dg.u = dgs.u
		dg.v = dgs.v + dv
		dg.nn = geo.normalize(geo.Normal.from_arr(dgs.dpdu.cross(dgs.dpdv) + dv * dgs.dndv))

		v_disp = d(dg)

		disp = d(dgs)

		# compute bump mapped dg
		dg = dgs.copy()

		# geo.Vector * geo.Normal -> geo.Vector
		dg.dpdu = dgs.dpdu + (u_disp - disp) / du * dgs.nn + disp * geo.Vector.fromNormal(dgs.dndu)
		dg.dpdv = dgs.dpdv + (v_disp - disp) / dv * dgs.nn + disp * geo.Vector.fromNormal(dgs.dndv)
		dg.nn = geo.Normal.from_arr(geo.normalize(dg.dpdu.cross(dg.dpdv)))
		if dgs.shape.ro ^ dgs.shape.transform_swaps_handedness:
			dg.nn *= -1.

		# orient shding normal to match goemtric normal
		dg.nn = geo.face_forward(dg.nn, dgg.nn)

		return dg


class MatteMaterial(Material):
	"""
	MatteMaterial Class

	Models prurely diffuse surfaces.

	Kd: Specular Diffuse Reflection Value
	sigma: Roughness. Returns Lambertian if
			0 at some point; OrenNayer otherwise

	"""
	def __init__(self, Kd: 'Texture', sigma: 'Texture', bump_map: 'Texture'=None):
		"""
		Kd: `Spectrum` `Texture`
		sigma, bump_map: `FLOAT` `Texture`
		"""
		self.Kd = Kd
		self.sigma = sigma
		self.bump_map = bump_map

	def get_bsdf(self, dg_g: 'geo.DifferentialGeometry', dg_s: 'geo.DifferentialGeometry') -> 'BSDF':
		from pytracer.reflection import (BSDF, Lambertian, OrenNayar)
		# possibly bump mapping
		if self.bump_map is not None:
			dgs = self.bump(self.bump_map, dg_g, dg_s)
		else:
			dgs = dg_s.copy()

		bsdf = BSDF(dgs, dg_g.nn)

		# evaluate textures
		r = util.clip(self.Kd(dgs))
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
	def __init__(self, Kd: 'Texture', Ks: 'Texture', roughness: 'Texture', bump_map: 'Texture'=None):
		"""
		Kd: `Spectrum` `Texture`
		sigma, bump_map: `FLOAT` `Texture`
		"""
		self.Kd = Kd
		self.Ks = Ks
		self.roughness = roughness
		self.bump_map = bump_map

	def get_bsdf(self, dg_g: 'geo.DifferentialGeometry', dg_s: 'geo.DifferentialGeometry') -> 'BSDF':
		from pytracer.reflection import (BSDF, Lambertian, FresnelDielectric, Microfacet, Blinn)
		# possibly bump mapping
		if self.bump_map is not None:
			dgs = Material.bump(self.bump_map, dg_g, dg_s)
		else:
			dgs = dg_s.copy()

		bsdf = BSDF(dgs, dg_g.nn)

		# diffuse
		kd = self.Kd(dgs).clip()
		diff = Lambertian(kd)
		fresnel = FresnelDielectric(Spectrum(1.5), Spectrum(1.))

		# glossy specular
		ks = self.Ks(dgs).clip()
		rough = self.roughness(dgs)
		spec = Microfacet(ks, fresnel, Blinn(1. / rough))

		bsdf.push_back(diff)
		bsdf.push_back(spec)

		return bsdf


class UberMaterial(Material):
	"""
	Ubse Class

	Models shinny metals (kitchen sink).

	Kd: Coef of diffuse reflection
	Ks: Coef of glossy reflection
	Kr: Coef of specular reflection
	Kt: Coef of specular transmission
	roughness: roughness
	eta: refraction of surface
	"""
	def __init__(self, Kd: 'Texture', Ks: 'Texture', Kr: 'Texture', Kt: 'Texture',
	             roughness: 'Texture', opacity: 'Texture', eta: 'Texture', bump_map: 'Texture'=None):
		"""
		Kd, Ks, Kr, Kt, opacity: `Spectrum` `Texture`
		roughness, eta, bump_map: `FLOAT` `Texture`
		"""
		self.Kd = Kd
		self.Ks = Ks
		self.Kr = Kr
		self.Kt = Kt
		self.opacity = opacity
		self.eta = eta
		self.roughness = roughness
		self.bump_map = bump_map

	def get_bsdf(self, dg_g: 'geo.DifferentialGeometry', dg_s: 'geo.DifferentialGeometry') -> 'BSDF':
		from pytracer.reflection import (BSDF, BDF, SpecularTransmission, SpecularReflection, Lambertian,
		                                 FresnelDielectric, Microfacet, Blinn)
		# possibly bump mapping
		if self.bump_map is not None:
			dgs = Material.bump(self.bump_map, dg_g, dg_s)
		else:
			dgs = dg_s.copy()

		bsdf = BSDF(dgs, dg_g.nn)
		op = self.opacity(dgs).clip()   # op: Spectrum

		if not (op == 1.).all():
			bsdf.push_back(SpecularTransmission((1. - op), 1., 1.))

		kd = op * self.Kd(dgs).clip()
		if not kd.is_black():
			bsdf.push_back(Lambertian(kd))

		e = self.eta(dgs)
		ks = op * self.Ks(dgs).clip()
		if not ks.is_black():
			fresnel = FresnelDielectric(e, 1.)
			rough = self.roughness(dgs)
			bsdf.push_back(Microfacet(ks, fresnel,Blinn(1. / rough)))

		kr = op * self.Kr(dgs).clip()
		if not kr.is_black():
			fresnel = FresnelDielectric(e, 1.)
			bsdf.push_back(SpecularReflection(kr, fresnel))

		kt = op * self.Kt(dgs).clip()
		if not kt.is_black():
			bsdf.push_back(SpecularTransmission(kt, e, 1.))

		return bsdf


class MixMaterial(Material):
	"""
	MixMaterial Class

	Models mixed materials. Use texture
	spectrum to blend.
	"""
	def __init__(self, m1: 'Material', m2: 'Material', scale: 'Texture'):
		"""
		m1, m2: `Material` `Spectrum`
		scale: `Spectrum` `Texture` for blend
		"""
		self.m1 = m1
		self.m2 = m2
		self.scale = scale

	def get_bsdf(self, dg_g: 'geo.DifferentialGeometry', dg_s: 'geo.DifferentialGeometry') -> 'BSDF':
		"""
		BSDF in m1 (i.e., `m1.bdfs`) is modified
		"""
		from pytracer.reflection import ScaledBDF

		b1 = self.m1.get_bsdf(dg_g, dg_s)
		b2 = self.m2.get_bsdf(dg_g, dg_s)

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
	def __init__(self, filename: 'str', bump_map: 'Texture'=None):
		"""
		filename: filename of measured material
		bump_map: `FLOAT` `Texture`

		Irregular Sampled Isotropic BRDF:
			*.brdt
		Regular Halfangle BRDF:
			*.merl (Binary)
		"""
		from pytracer.reflection import (brdf_remap, IrIsotropicBRDFSample)

		self.bump_map = bump_map
		self.regular_data = None
		self.theta_phi_tree = None
		self.theta_phi_samples = None

		dire, file = os.path.split(filename)
		name, ext = os.path.splitext(file)

		# Irregular Sampled Isotropic BRDF
		if ext == '.brdf' or ext == '.BRDF':
			if file in IrIsotropicData:
				# already read
				self.theta_phi_sample, self.theta_phi_tree = IrIsotropicData[file]
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
			tree_data = np.empty([num_smp, 3])  # indexed by point structure

			pos = 1 + num_wls
			cnt = 0
			while pos < len_val:
				thi, phi, tho, pho = val[pos:pos+4]
				pos += 4
				wo = geo.spherical_direction(np.sin(tho), np.cos(tho), pho)
				wi = geo.spherical_direction(np.sin(thi), np.cos(thi), phi)
				s = Spectrum.from_sampled(wls, val[pos:pos + num_wls])
				p = brdf_remap(wo, wi)
				smp = IrIsotropicBRDFSample(p, s)
				tree_data[cnt, :] = smp.p
				samples.append(smp)
				pos += num_wls
				cnt += 1

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

	def get_bsdf(self, dg_g: 'geo.DifferentialGeometry', dg_s: 'geo.DifferentialGeometry') -> 'BSDF':
		"""
		BSDF in m1 (i.e., `m1.bdfs`) is modified
		"""
		from pytracer.reflection import (BSDF, ReHalfangleBRDF, IrIsotropicBRDF)
		if self.bump_map is not None:
			dgs = Material.bump(self.bump_map, dg_g, dg_s)
		else:
			dgs = dg_s.copy()

		bsdf = BSDF(dgs, dg_g.nn)

		if self.regular_data is not None:
			# regular tabulate
			bsdf.push_back(ReHalfangleBRDF(self.regular_data, self.n_theta_h, self.n_theta_d, self.n_phi_d))

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
					sigma_prime_s: 'Texture', eta: 'Texture', bump_map: 'Texture'=None):
		self.scale = scale
		self.kr = kr
		self.sigma_a = sigma_a
		self.sigma_prime_s = sigma_prime_s
		self.eta = eta
		self.bump_map = bump_map

	def get_bsdf(self, dg_g: 'geo.DifferentialGeometry', dg_s: 'geo.DifferentialGeometry') -> 'BSDF':
		from pytracer.reflection import (BSDF, SpecularReflection, FresnelDielectric)

		if self.bump_map is not None:
			dgs = Material.bump(self.bump_map, dg_g, dg_s)
		else:
			dgs = dg_s.copy()
		R = self.kr(dgs).clip()
		bsdf = BSDF(dgs, dg_g.nn)

		if not R.is_black():
			bsdf.push_back(SpecularReflection(R, FresnelDielectric(Spectrum(1.), self.eta(dgs))))

		return bsdf

	def get_bssrdf(self, dg_g: 'geo.DifferentialGeometry', dg_s: 'geo.DifferentialGeometry') -> 'BSSRDF':
		"""
		get_bssrdf()

		Specified in terms of reciprocal
		distance ($m^{-1}$).
		"""
		from pytracer.volume import BSSRDF
		return BSSRDF(self.scale * self.sigma_a(dg_s), self.scale * self.sigma_prime_s(dg_s), self.eta(dg_s))


class KdSubsurfaceMaterial(Material):
	"""
	SubsurfaceMaterial Class

	Models translucent objects
	"""
	def __init__(self, kd: 'Texture', kr: 'Texture', mfp: 'Texture',
					eta: 'Texture', bump_map: 'Texture'=None):
		self.kd = kd
		self.kr = kr
		self.mfp = mfp # mean free path
		self.eta = eta
		self.bump_map = bump_map

	def get_bsdf(self, dg_g: 'geo.DifferentialGeometry', dg_s: 'geo.DifferentialGeometry') -> 'BSDF':
		from pytracer.reflection import (BSDF, SpecularReflection, FresnelDielectric)

		if self.bump_map is not None:
			dgs = Material.bump(self.bump_map, dg_g, dg_s)
		else:
			dgs = dg_s.copy()

		R = self.kr(dgs).clip()
		bsdf = BSDF(dgs, dg_g.nn)

		if not R.is_black():
			bsdf.push_back(SpecularReflection(R, FresnelDielectric(Spectrum(1.), self.eta(dgs))))

		return bsdf

	def get_bssrdf(self, dg_g: 'geo.DifferentialGeometry', dg_s: 'geo.DifferentialGeometry') -> 'BSSRDF':
		"""
		get_bssrdf()

		Specified in terms of reciprocal
		distance ($m^{-1}$).
		"""
		from pytracer.volume import (BSSRDF, subsurface_from_diffuse)
		e = self.eta(dg_s)
		sigma_a, sigma_prime_s = subsurface_from_diffuse(self.kd(dg_s).clip(), self.mfp(dg_s), e)

		return BSSRDF(sigma_a, sigma_prime_s, e)

