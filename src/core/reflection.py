'''
reflection.py

Model distribution functions.
Convention:
	Incident light and viewing direction
	are normalized and face outwards;
	Normal faces outwards and is not
	flipped to lie in the same side as
	viewing direction.

Created by Jiayao on Aug 2, 2017
'''
from numba import jit
from abc import ABCMeta, abstractmethod
import numpy as np
from src.core.pytracer import *
from src.core.geometry import *
from src.core.diffgeom import *
from src.core.spectrum import *
from src.core.sampler import stratified_sample_2d	# used in BSDF
from src.core.montecarlo import *

# inline functions
def cos_theta(w: 'Vector'): return w.z
def abs_cos_theta(w: 'Vector'): return np.fabs(w.z)
def sin_theta_sq(w: 'Vector'): return max(0., 1. - w.z * w.z)
def sin_theta(w: 'Vector'): return np.sqrt(max(0., 1. - w.z * w.z))
@jit
def cos_phi(w: 'Vector'): return 1. if max(0., 1. - w.z * w.z) == 0. \
								 else np.clip(w.x / np.sqrt(max(0., 1. - w.z * w.z)), -1., 1.)

@jit
def sin_phi(w: 'Vector'): return 0. if max(0., 1. - w.z * w.z) == 0. \
								 else np.clip(w.y / np.sqrt(max(0., 1. - w.z * w.z)), -1., 1.)

# utility functions
@jit
def fr_diel(cos_i: FLOAT, cos_t: FLOAT, eta_i: 'Spectrum', eta_t: 'Spectrum') -> 'Spectrum':
	'''
	fr_diel

	Compute reflectance using Fresnel formula for
	dielectric materials and circularly polarized light
	'''
	r_para = ((eta_t * cos_i ) - (eta_i * cos_t)) / \
				((eta_t * cos_i) + (eta_t * cos_t))
	r_perp = ((eta_i * cos_i) - (eta_t * cos_t)) / \
				((eta_i * cos_i) + (eta_t * cos_t))
	return .5 * (r_para * r_para + r_perp * r_perp)

@jit
def fr_cond(cos_i: FLOAT, eta: 'Spectrum', k: 'Spectrum') -> 'Spectrum':
	'''
	fr_cond

	Compute approx. reflectance using Fresnel formula for
	conductor materials and circularly polarized light
	'''
	tmp = (eta * eta + k * k) * cos_i * cos_i
	r_para_sq = (tmp - (2. * eta * cos_i) + 1.) / \
				(tmp + (2. * eta * cos_i) + 1.)
	tmp = eta * eta + k * k
	r_perp_sq = (tmp - (2. * eta * cos_i) + cos_i * cos_i) / \
				(tmp + (2. * eta * cos_i) + cos_i * cos_i)
	return .5 * (r_para_sq + r_perp_sq)

@jit
def BRDF_remap(wo: 'Vector', wi: 'Vector') -> 'Point':
	'''
	Mapping regularly sampled BRDF
	using Marschner, 1998
	'''
	dphi = spherical_phi(wi) - spherical_phi(wo)
	if dphi < 0.:
		dphi += 2. * PI
	if dphi > 2. * PI:
		dphi -= 2. * PI
	if dphi > PI:
		dphi = 2. * PI - dphi

	return Point(sin_theta(wi) * sin_theta(wo),
				dphi * INV_PI, cos_theta(wi) * cos_theta(wo))


# interface for Fresnel coefficients
class Fresnel(object, metaclass=ABCMeta):
	'''
	Fresnel Class

	Interface for computing Fresnel coefficients
	'''

	@abstractmethod
	def __call__(self, cosi: FLOAT):
		raise NotImplementedError('src.core.reflection.{}.__call__: abstract method '
									'called'.format(self.__class__))

class FresnelConductor(Fresnel):
	'''
	Fresnel Class

	Implement Fresnel interface for conductors
	'''
	def __init__(self, eta: 'Spectrum', k: 'Spectrum'):
		self.eta = eta
		self.k = k
	def __call__(self, cosi: FLOAT):
		return fr_cond(np.fabs(cosi), self.eta, self.k)

class FresnelDielectric(Fresnel):
	'''
	Fresnel Class

	Implement Fresnel interface for conductors
	'''
	def __init__(self, eta_i: 'Spectrum', eta_t: 'Spectrum'):
		self.eta_i = eta_i
		self.eta_t = eta_t

	def __call__(self, cosi: FLOAT):
		ci = np.clip(cosi, -1., 1.)
		# indices of refraction
		ei = self.eta_i
		et = selt.eta_t
		if cosi < 0.:
			# ray is on the inside
			ei, et = et, ei

		# Snell's law
		st = ei / et * np.sqrt(max(0., 1. - ci * ci))

		if st >= 1.:
			# total internal reflection
			return 1.

		else:
			ct = np.sqrt(max(0., 1. - st * st))
			return fr_diel(np.fabs(ci), ct, ei, et)

class FresnelFullReflect(Fresnel):
	'''
	FresnelFullReflect Class

	Return full `Spectrum` for all
	incoming directions
	'''
	def __call__(self, cosi: FLOAT):
		return Spectrum(1.)

# interface for distribution functions
# class BDFType(IntEnum):
# 	pass

class BDFType():
	'''
	Wrapper for
	integer enumeration with
	bitmask ops
	'''
	def __init__(self, v):
		if isinstance(v, BDFType):
			self.v = v.v
		else:
			self.v = UINT(v)	# raise exception if needed
	def __repr__(self):
		return "{}\nEnum: {}".format(self.__class__, self.v)

	def __invert__(self): return BDFType(~self.v)
	def __or__(self, other): return BDFType(self.v | other.v) if isinstance(other, BDFType) \
									else BDFType(self.v | UINT(other))
	def __and__(self, other): return BDFType(self.v & other.v) if isinstance(other, BDFType) \
									else BDFType(self.v & UINT(other))
	def __xor__(self, other): return BDFType(self.v ^ other.v) if isinstance(other, BDFType) \
									else BDFType(self.v ^ UINT(other))
	def __lshift__(self, other): return BDFType(self.v << other)
	def __rshift__(self, other): return BDFType(self.v >> other)

	REFLECTION = (1 << 0)
	TRANSMISSION = (1 << 1)
	DIFFUSE = (1 << 2)
	GLOSSY = (1 << 3)
	SPECULAR = (1 << 4)
	ALL_TYPES = (DIFFUSE | GLOSSY | SPECULAR)
	ALL_REFLECTION = (REFLECTION | \
					 	ALL_TYPES)
	ALL_TRANSMISSION = (TRANSMISSION | \
					   ALL_TYPES)
	ALL = (ALL_REFLECTION | ALL_TRANSMISSION)



class BDF(object, metaclass=ABCMeta):
	'''
	BDF Class

	Models bidirectional distribution function.
	Base class for `BSDF` and `BRDF`
	'''

	def __init__(self, t: 'BDFType'):
		self.type = BDFType(t)

	def __repr__(self):
		return "{}\nType: {}".format(self.__class__, self.type.v)


	@abstractmethod
	def f(self, wo: 'Vector', wi: 'Vector') -> 'Spectrum':
		'''
		Returns the BDF for given pair of directions
		Asssumes light at different wavelengths are
		decoupled.
		'''
		raise NotImplementedError('src.core.reflection.{}.f(): abstract method '
									'called'.format(self.__class__)) 		

	def pdf(self, wo: 'Vector', wi: 'Vector') -> FLOAT:
		'''
		pdf()

		Returns pdf of given direction
		'''
		if wo.z * wi.z > 0.:
			# same hemisphere
			return abs_cos_theta(wi) * INV_PI
		return 0.


	def sample_f(self, wo: 'Vector', u1: FLOAT, 
							u2: FLOAT) -> [FLOAT, 'Vector', 'Spectrum']:
		'''
		Handles scattering discribed by delta functions
		or random sample directions.
		Returns the spectrum, incident vector and pdf used in MC sampling.

		By default samples from a hemeisphere with
		cosine-wighted distribution.

		Returns:
		[pdf, wi, Spectrum]
		'''
		# cosine sampling
		wi = cosine_sample_hemishpere(u1, u2)
		if wo.z < 0.:
			wi.z *= -1.

		return [self.pdf(wo, wi), self.f(wo, wi), wi]

	@jit
	def rho_hd(self, w: 'Vector', samples: [FLOAT]) -> 'Spectrum':
		'''
		Computs hemispherical-directional reflectance function.

		- w
			Incoming 'Vector'
		- samples
			2d np array
		'''
		r = Spectrum(0.)
		for smp samples:
			wi, pdf, f = self.sample_f(w, smp[0], smp[1])
			if pdf > 0.:
				r += f * abs_cos_theta(wi) / pdf

		r /= nSamples
		return r

	def rho_hh(self, nSamples: INT, samples_1: [FLOAT], samples_2: [FLOAT]) -> 'Spectrum':
		'''
		Computs hemispherical-hemispherical reflectance function.

		- samples_1, samples_2
			2d np array
		'''
		r = Spectrum(0.)
		for i in range(nSamples):
			wo = uniform_sample_hemisphere(samples_1[i][0], samples_1[i][1])
			pdf_o = INV_2PI

			pdf_i, wi, f = self.sample_f(wo, samples_2[i][0], samples_2[i][1])

			if pdf_i > 0.:
				r += f * abs_cos_theta(wi) * abs_cos_theta(wo) / (pdf_o * pdf_i)

		r /= nSamples
		return r

	def match_flag(self, flag: 'BDFType') -> bool:
		return (self.type & flag) == type




# adapter from BRDF to BTDF
class BRDF2BTDF(BDF):
	'''
	BRDF2BTDF Class

	Adpater class to convert a BRDF to
	BTDF by flipping incident light direction
	and forward function calls to BRDF
	'''
	def __init__(self, b: 'BDF'):
		super().__init__(b.type ^ (BDFType.REFLECTION | BDFType.TRANSMISSION))
		self.brdf = b

	@staticmethod
	def switch(w: 'Vector'):
		return Vector(w.x, w.y, -w.z)

	# forward function calls
	def f(self, wo: 'Vector', wi: 'Vector'): return self.brdf.f(wo, self.switch(wi))	
	def sample_f(self, wo: 'Vector', u1: FLOAT, 
					u2: FLOAT): return self.brdf.sample_f(wo, self.switch(wi),
															u1, u2, pdf)	
	def rho_hd(self, wo: 'Vector', samples: [FLOAT]): return self.brdf.rho_hd(wo, samples)
	def rho_hh(self, nSamples: INT, samples_1: [FLOAT],
					samples_2: [FLOAT]): return self.brdf.rho_hh(nSamples, samples_1, samples_2)

# adapter for scaling BDF
class ScaledBDF(BDF):
	'''
	ScaledBDF Class

	Wrapper for scaling BDF based on
	given `Spectrum`
	'''
	def __init__(self, b: 'BDF', sc: 'Spectrum'):
		super().__init__(b.type)
		self.bdf = b
		self.s = sc

	# scale by spectrum
	def f(self, wo: 'Vector', wi: 'Vector'): return self.s * self.brdf.f(wo, wi)	
	def sample_f(self, wo: 'Vector', u1: FLOAT, 
					u2: FLOAT): return self.s * self.brdf.sample_f(wo, wi,
															u1, u2, pdf)	
	def rho_hd(self, wo: 'Vector', samples: [FLOAT]): return self.s * self.brdf.rho_hd(wo, samples)
	def rho_hh(self, nSamples: INT, samples_1: [FLOAT],
					samples_2: [FLOAT]): return self.s * self.brdf.rho_hh(nSamples, samples_1, samples_2)

class SpecularReflection(BDF):
	'''
	SpecularReflection Class

	Models perfect specular reflection.
	'''
	def __init__(self, sp: 'Spectrum', fr: 'Fresnel'):
		super().__init__(BDFType.REFLECTION | BDFType.SPECULAR)
		self.R = sp
		self.fresnel = fr

	def f(self, wo: 'Vector', wi: 'Vector') -> 'Spectrum':
		'''
		Return 0., singularity at perfect reflection
		will be handled in light transport routines.
		'''
		return Spectrum(0.)

	def pdf(self, wo: 'Vector', wi: 'Vector') -> FLOAT:
		return 0.

	def sample_f(self, wo: 'Vector', u1: FLOAT, 
							u2: FLOAT) -> [FLOAT, 'Vector', 'Spectrum']:
		# find direction
		wi = Vector(-wo.x, -wo.y, wz)

		return [1., wi, self.fresnel(cos_theta(wo)) * self.R / abs_cos_theta(wi)] # 1. suggests no MC samples needed



class SpecularTransmission(BDF):
	'''
	SpecularTransmission Class

	Models specular transmission
	using delta functions
	'''
	def __init__(self, t: 'Spectrum', ei: FLOAT, et: FLOAT):
		super().__init__(BDFType.TRANSMISSION | BDFType.SPECULAR)
		self.fresnel = FresnelDielectric(ei, et)	# conductors do not transmit light
		self.T = t
		self.ei = ei
		self.et = et

	def f(self, wo: 'Vector', wi: 'Vector') -> 'Spectrum':
		'''
		Return 0., singularity at perfect reflection
		will be handled in light transport routines.
		'''
		return Spectrum(0.)

	def pdf(self, wo: 'Vector', wi: 'Vector') -> FLOAT:
		return 0.

	@jit
	def sample_f(self, wo: 'Vector', u1: FLOAT, 
							u2: FLOAT) -> [FLOAT, 'Vector', 'Spectrum']:
		# find eta pair
		ei, et = self.ei, self.et
		if cos_theta(wo) > 0.:
			ei, et = et, ei

		# compute transmited ray direction
		si_sq = sin_theta_sq(wo)
		eta = ei / et
		st_sq = eta * eta * si_sq

		if st_sq >= 1.:
			return [0., None, None]

		ct = np.sqrt(max(0., 1. - st_sq))
		if cos_theta(wo) > 0.:
			ct = -ct
		wi = Vector(eta * (-wo.x), eta * (-wo.y), ct)


		F = self.fresnel(cos_theta(wo.t))

		return [1., wi, (et * et) / (ei * ei) * (Spectrum(1.) - F) * \
					 self.T / abs_cos_theta(wi)] # 1. suggests no MC samples needed

class Lambertian(BDF):
	'''
	Lambertian Class

	Models lambertian.
	'''
	def __init__(self, r: 'Spectrum'):
		'''
		R: Spectrum Reflectance
		'''
		super().__init__(BDFType.REFLECTION | BDFType.DIFFUSE)
		self.R = r

	def f(self, wo: 'Vector', wi: 'Vector') -> 'Spectrum':
		'''
		Return 0., singularity at perfect reflection
		will be handled in light transport routines.
		'''
		return INV_PI * self.R

	def rho_hd(self, wo: 'Vector', samples: [FLOAT]) -> 'Spectrum':
		return self.R


	def rho_hh(self, nSamples: INT, samples_1: [FLOAT], samples_2: [FLOAT]) -> 'Spectrum':
		return self.R



# Oren Nayer Model for Rough Surfaces
class OrenNayar(BDF):
	'''
	OrenNayar Class

	Using Oren-Nayer approximation
	for surface scattering:
	$$
	f_r( \omega_i, \omega_o ) = \frac{\pho}{\pi} (A + B \max(0, \
		\cos(phi_i - \phi_o)) * \sin \alpha \tan \beta )
	$$
	where
	$$
	\alpha = \max(\theta_i, \theta_o) \\
	\beta = \min(\theta_i, \theta_o)
	$$
	'''
	def __init__(self, r: 'Spectrum', sig: FLOAT):
		super().__init__(BDFType.REFLECTION | BDFType.DIFFUSE)
		self.R = r
		sig = np.deg2rad(sig)
		sig_sq = sig * sig
		self.A = 1. - (sig_sq / (2. * (sig_sq + 0.33)))
		self.B = .45 * sig_sq / (sig_sq + .09)

	@jit
	def f(self, wo: 'Vector', wi: 'Vector') -> 'Spectrum':
		st_i = sin_theta(wi)
		st_o = sin_theta(wo)

		# \cos(\phi_i - \phi_o)
		mc = 0.
		if st_i > EPS and st_o > EPS:
			dcos = cos_phi(wi) * cos_phi(wo) + sin_phi(wi) * sin_phi(wo)
			mc = max(0., dcos)

		# \sin \alpah \tan \beta
		if abs_cos_theta(wi) > abs_cos_theta(wo):
			sa = st_o
			tb = st_i / abs_cos_theta(wi)
		else:
			sa = st_i
			tb = st_o / abs_cos_theta(wo)


		return self.R * INV_PI * (self.A + self.B * mc * sa * tb)



# Torrance–Sparrow Model
class MicrofacetDistribution(object, metaclass=ABCMeta):
	'''
	MicrofacetDistribution Class

	Compute microfacet distribution using
	Torrance–Sparrow model.
	Microfacet distribution functions must
	be normalized.
	'''

	def __repr__(self):
		return "{}\n".format(self.__class__)	

	@abstractmethod
	def D(self, wh: 'Vector') -> FLOAT:
		raise NotImplementedError('src.core.reflection.{}.D(): abstract method '
									'called'.format(self.__class__)) 

	@abstractmethod
	def sample_f(self, wo: 'Vector', u1: FLOAT, 
							u2: FLOAT) -> [FLOAT, 'Vector', 'Spectrum']:
		raise NotImplementedError('src.core.reflection.{}.sample_f(): abstract method '
									'called'.format(self.__class__)) 

	@abstractmethod
	def pdf(self, wo: 'Vector', wi: 'Vector') -> FLOAT:
		raise NotImplementedError('src.core.reflection.{}.pdf(): abstract method '
									'called'.format(self.__class__)) 

# Blinn Model
class Blinn(MicrofacetDistribution):
	'''
	Blinn Class

	Models Blinn microfacet distribution:
	$$
	D(\omega_h) \prop (\omega_h \cdot \mathbf(n)) ^ {e}
	$$
	Apply normalization constraint:
	$$
	D(\omega_h) = \frac{e+2}{2\pi} (\omega_h \cdot \mathbf{n}) ^ {e}
	$$
	'''
	def __init__(self, e: FLOAT):
		self.e = np.clip(e, -np.inf, 10000.)

	@jit
	def D(self, wh: 'Vector') -> FLOAT:
		return (self.e + 2) * INV_2PI * np.power(abs_cos_theta(wh), self.e)

	def sample_f(self, wo: 'Vector', u1: FLOAT, 
							u2: FLOAT) -> [FLOAT, 'Vector']:
		# compute sampled half-angle vector
		ct = np.power(u1, 1. / (self.e + 1))
		st = np.sqrt(max(0., 1. - ct * ct))
		phi = u2 * 2. * PI
		wh = spherical_direction(st, ct, phi)

		if wo.z * wh.z <= 0.:
			wh *= -1.

		# incident direction by reflection
		wi = -wo + 2. * wo.dot(wh) * wh

		# pdf
		pdf = ((self.e + 1.) * np.ower(ct, self.e)) / \
				(2. * PI * 4. * wo.dot(wh))

		if wo.dot(wh) <= 0.:
			pdf = 0.

		return [pdf, wi]


	def pdf(self, wo: 'Vector', wi: 'Vector') -> FLOAT:
		raise NotImplementedError('src.core.reflection.{}.pdf(): abstract method '
									'called'.format(self.__class__)) 

# Anisotropic, Ashikhmin and Shirley
class Anisotropic(MicrofacetDistribution):
	def __init__(self, ex: FLOAT, ey: FLOAT):
		self.ex = np.clip(ex, -np.inf, 10000.)
		self.ey = np.clip(ey, -np.inf, 10000.)
	@jit
	def D(self, wh: 'Vector') -> FLOAT:
		cth = abs_cos_theta(wh)
		d = 1. - cth * cth
		
		if d == 0.:
			return 0.

		e = (self.ex * wh.x * wh.x + self.ey * wh.y * wh.y) / d	
		return np.sqrt((self.ex + 2.) * (self.ey + 2.)) * INV_2PI * np.power(cth, e)

	def __sample_first_quad(self, u1: FLOAT, u2: FLAOT) -> [FLOAT, FLOAT]:
		'''
		__sample_first_quad()

		Samples a direction in the first quadrant of
		unit hemisphere. Returns [phi, cos(theta)]
		'''
		if self.ex == self.ey:
			phi = PI * u1 * .5
		else:
			phi = np.arctan(np.sqrt((self.ex + 1.) / (self.ey + 1.)) * np.tan(PI * u1 * .5))

		cp = np.cos(phi)
		sp = np.sin(phi)

		return [phi, np.power(u2, 1. / (self.ex * cp * cp + self.ey * sp * sp + 1.))]

	def sample_f(self, wo: 'Vector', u1: FLOAT, 
							u2: FLOAT) -> [FLOAT, 'Vector']:
		# sample from first quadrant and remap to hemisphere to sample w_h
		if u1 < .25:
			phi, ct = self.__sample_first_quad(4. * u1, u2)
		elif u1 < .5:
			u1 = 4. * (.5 - u1)
			phi, ct = self.__sample_first_quad(u1, u2)
			phi = PI - phi
		elif u1 < .75:
			u1 = 4. * (u1 - .5)
			phi, ct = self.__sample_first_quad(u1, u2)
			phi += PI
		else:
			u1 = 4. * (1. - u1)
			phi, ct = self.__sample_first_quad(u1, u2)
			phi = 2. * PI - phi

		st = np.sqrt(max(0., 1. - ct * ct))
		wh = spherical_direction(st, ctm phi)
		if wo.z * wh.z <= 0.:
			wh *= -1.

		# incident direction by reflection
		wi = -wo + 2. * wo.dot(wh) * wh

		# compute pdf for w_i
		ct = abs_cos_theta(wh)
		ds = 1. - ct * ct
		if ds > 0. and wo.dot(wh) > 0.:
			return [(np.sqrt((self.ex + 1.) * (self.ey + 1.)) * INV_2PI * np.power(ct, 
								(self.ex * wh.x * wh.x + self.ey * wh.y * wh.y) / ds)) / \
						(4. * wo.dot(wh)), wi]
		else:
			return [0., wi]

	def pdf(self, wo: 'Vector', wi: 'Vector') -> FLOAT:
		wh = normalize(wo + wi)
		ct = abs_cos_theta(wh)
		ds = 1. - ct * ct
		if ds > 0. and wo.dot(wh) > 0.:
			return (np.sqrt((self.ex + 1.) * (self.ey + 1.)) * INV_2PI * np.power(ct, 
								(self.ex * wh.x * wh.x + self.ey * wh.y * wh.y) / ds)) / \
						(4. * wo.dot(wh))
		else:
			return 0.
class Microfacet(BDF):
	'''
	Microfacet Class

	Models microfaceted surface
	'''
	def __init__(self, r: 'Spectrum', f: 'Fresnel', d: 'MicrofacetDistribution'):
		self.R = r
		self.fresnel = f
		self.distribution = d

	@jit
	def f(self, wo: 'Vector', wi: 'Vector') -> 'Spectrum':
		ct_i = cos_theta(wi)
		ct_o = cos_theta(wo)

		if ct_i == 0. or ct_o == 0.:
			return Spectrum(0.)

		wh = normalize(wi + wo)
		ct_h = wi.dot(wh)
		F = self.fresnel(ct_h)

		return self.R * self.distribution.D(wh) * self.G(wo, wi, wh) * \
					F / (4. * ct_i * ct_o)

	@jit
	def G(self, wo: 'Vector', wi: 'Vector', wh: 'Vector') -> FLOAT:
		return min(1., min( (2. * abs_cos_theta(wh) * abs_cos_theta(wo) / wo.abs_dot(wh)),
							(2. * abs_cos_theta(wh) * abs_cos_theta(wi) / wo.abs_dot(wh))))

	def sample_f(self, wo: 'Vector', u1: FLOAT, 
							u2: FLOAT) -> [FLOAT, 'Vector', 'Spectrum']:
		[pdf, wi, spec] = self.distribution.sample_f(wo, u1, u2)
		if wi.z * wo.z <= 0.:
			return [pdf, wi, Spectrum(0.)]
		return [pdf, wi, self.f(wo, wi)]

	def pdf(self, wo: 'Vector', wi: 'Vector') -> FLOAT:
		if wi.z * wo.z <= 0.:
			return 0.
		return self.distribution.pdf(wo, wi)





# Fresnel Blend Model, Ashikhmin and Shirley
# Account for, e.g., glossy on diffuse
class FresnelBlend(BDF):
	'''
	FresnelBlend Class

	Based on the weighted sum of
	glossy and diffuse term.
	'''
	def __init__(self, d: 'Spectrum', s: 'Spectrum', dist : 'MicrofacetDistribution'):
		super().__init__(BDFType.REFLECTION | BDFType.GLOSSY)
		self.Rd = d
		self.Rs = s
		self.distribution = dist

	@jit
	def schlick(self, ct: FLOAT) -> 'Spectrum':
		'''
		Schlick (1992) Approximation
		of Fresnel reflection
		'''
		return self.Rs + np.power(1. - ct, 5.) * (Spectrum(1.) - self.Rs)

	def sample_f(self, wo: 'Vector', u1: FLOAT, 
							u2: FLOAT) -> [FLOAT, 'Vector', 'Spectrum']:
		if u1 < .5:
			u1 = 2. * u1
			# cosine sample the hemisphere
			wi = cosine_sample_hemishpere(u1, u2)
			if wo.z < 0.:
				wi.z *= -1.
		else:
			u1 = 2. * (u1 - .5)
			pdf, wi = self.distribution.sample_f(wo, u1, u2)
			if wo.z * wi.z <= 0.:
				# not on the same hemisphere
				return [pdf, wi, Spectrum(0.)]
		return [self.pdf(wo, wi), wi, self.f(wo, wi)]

	def pdf(self, wo: 'Vector', wi: 'Vector') -> FLOAT:
		if wo.z * wi.z <= 0.:
			return 0.
		return .5 * (abs_cos_theta(wi) * INV_PI + self.distribution.pdf(wo, wi))
	@jit
	def f(self, wo: 'Vector', wi: 'Vector') -> 'Spectrum':
		diffuse = (28. / (23. * PI)) * self.Rd * \
			(Spectrum(1.) - self.Rs) * \
			(1. - np.power(1. - .5  * abs_cos_theta(wi), 5)) * \
			(1. - np.power(1. - .5  * abs_cos_theta(wo), 5))

		wh = normalize(wi + wo)
		
		specular = self.distribution.D(wh) / \
			(4. * wi.abs_dot(wh) * max(abs_cos_theta(wi), abs_cos_theta(wo))) * \
			self.schlick(wi.dot(wh))

		return diffuse + specular

# Measured BDF
class IrIsotropicBRDFSample(object):
	'''
	IrIsotropicBRDFSample Class

	Store irregular isotropic BRDF sample
	'''
	def __init__(self, p: 'Point', v: 'Spectrum'):
		self.p = p.copy()
		self.v = v.copy()

	def __repr__(self):
		return "{}\nPoint: {}\nSpectrum:\n{}".format(self.__class__,
							self.p, self.v)


class IrIsotropicBRDF(BDF):
	'''
	IrIsotropicBRDF Class

	Used for measured BRDF
	'''
	def __init__(self, tree: 'KdTree', data: ['IrIsotropicBRDFSample']):
		super().__init__(BDFType.REFLECTION | BDFType.GLOSSY)
		self.tree = tree
		self.data = data


	@jit
	def f(self, wo: 'Vector', wi: 'Vector') -> 'Spectrum':
		m = BRDF_remap(wo, wi)
		max_dist = .01
		while True:
			indices = tree.query_ball_point(m, max_dist)
			if len(indices) > 2 or max_dist > 1.5:
				sum_wt = 0.
				v = Spectrum()
				for sample in self.data[idx]:
					wt += np.exp((sample.p - m).sq_length() * -100.)
					sum_wt += wt
					v += sample.v * wt

				return v.clip() / sum_wt 

			max_dist *= 1.414


class ReHalfangleBRDF(BDF):
	'''
	ReHalfangleBRDF Class

	Models regularly tabulated BRDF Samples
	Format inline with Matusik(2003) at
	http://people.csail.mit.edu/wojciech/BRDFDatabase/
	'''
	def __init__(self, data: 'np.ndarray', nth: INT, ntd: INT, npd: INT):
		super().__init__(BDFType.REFLECTION | BDFType.GLOSSY)
		self.brdf = data
		self.nThetaH = INT(nth)
		self.nThetaD = INT(ntd)
		self.nPhiD = INT(npd)

	@jit
	def f(self, wo: 'Vector', wi: 'Vector') -> 'Spectrum':
		# find wh and transform wi to halfangle coord. system
		wh = normalize(wi + wo)
		
		t = spherical_theta(wh)
		cp = cos_phi(wh)
		sp = sin_phi(wh)
		ct = cos_theta(wh)
		st = sin_theta(wh)
		x = Vector(cp * ct, sp * ct, -st)
		y = Vector(-sp, cp, 0.)
		wd = Vector(wi.dot(x), wi.dot(y), wi.dot(wh))

		# compute index
		wdt = spherical_theta(wd)
		wdp = spherical_phi(wd)
		if wdp > PI:
			wdp -= PI
		wht_idx = np.clip(INT(( np.sqrt(max(0., t / (PI/2.))) )/(self.nThetaH)), 0, self.nThetaH - 1)

		wdt_idx = np.clip(INT((wdt)/(self.nThetaD * PI)), 0, self.nThetaD - 1)

		wdp_idx = np.clip(INT((wdp)/(self.nPhiD * PI)), 0, self.nPhiD - 1)
		return Spectrum.fromRGB(self.brdf[wht_idx][wdt_idx][wdp_idx])


class BSDFSample(object):
	'''
	BSDFSample Class

	Encapsulate samples used
	by BSDF.sample_f()
	'''
	def __init__(self, up0: FLOAT=0., up1: FLOAT=0., u_com: FLOAT=0.):
		self.u_dir = [up0, up1]
		self.u_com = u_com

	def __repr__(self):
		return "{}\nDirection: ({},{})\nComponent: {}\n".format(self.__class__, self.u_dir[0], self.u_dir[1], self.u_com)

	@classmethod
	def fromRand(cls, rng=np.random.rand):
		self = cls(rng(), rng(), rng())
		return self

	@classmethod
	def fromSample(cls, sample: 'Sample', offset: 'BSDFSampleOffset', n: UINT):
		self = cls(sample.twoD[offset.offset_dir][n][0],
				   sample.twoD[offset.offset_dir][n][1],
				   sample.oneD[offset.offset_com][n])
		return self

class BSDFSampleOffset(object):
	'''
	BSDFSampleOffset Class

	Encapsulate offsets provided
	by sampler
	'''
	def __init__(self, nSamples: INT, sample: 'Sample'):
		self.nSamples = nSamples
		self.offset_com = sample.add_1d(nSamples)
		self.offset_dir = sample.add_2d(nSamples)

	def __repr__(self):
		return "{}\nSamples: {}\n".format(self.__class__, self.nSamples)


class BSDF(object):
	'''
	BSDF Class

	Models the collection of BRDF and BTDF
	Also responsible for shading Normals.

	n_s: shading normal
	n_g: geometric normal
	'''
	def __init__(self, dg: 'DifferentialGeometry', ng: 'Normal', e: FLOAT=1.):
		'''
		dg: DifferentialGeometry
		ng: Geometric Normal
		e: index of refraction
		'''
		self.dgs = dg
		self.eta = e
		self.ng = ng

		# coord. system
		self.nn = dg.nn
		self.sn = normalize(dg.dpdu)
		self.tn = nn.cross(sn)

		self.bdfs = []
		self.__nBDF = INT(0)

	def __repr__(self):
		return "{}\nBDF Count: {}\nPoint: {}".format(self.__class__,
						self.nBDF, self.dgs.p)
	@jit
	def sample_f(self, wo_w: 'Vector', bsdf_smp: 'BSDFSample',
				flags: 'BDFType') -> [FLOAT, 'Vector', 'BDFType' 'Spectrum']:
		'''
		sample_f()

		returns [pdf, wi_w, sample_type, spectrum]
		'''
		# choose BDFs
		smp_type = None
		n_match = self.n_components(flags)
		if n_match == 0:
			return [0., None, None, Spectrum(0.)]

		cnt = last = min(ftoi(bsdf_smp.u_com * n_match), n_match - 1)
		for func in self.bdfs:
			if func.match_flag(flags):
				cnt -= 1
				if cnt == 0:
					bdf = func
					break
		# sample BDFs
		wo = self.w2l(wo_w)
		pdf, wi, f = bdf.sample_f(wo, bsdf_smp.u_dir[0], bsdf_smp.u_dir[1])
		wi_w = self.l2w(wi)

		if pdf == 0.:
			return [pdf, wi_w, bdf.type, Spectrum(0.)]

		# compute overall pdf
		if (not (bdf.type & BDFType.SPECULAR)) and n_match > 1:
			for func in self.bdfs:
				if func != bdf and func.match_flag(flags):
					pdf += func.pdf(wo, wi)

		if n_match > 1:
			pdf /= n_match

		# compute value of BSDF for sampled direction
		if not bdf.type & BDFType.SPECULAR:
			f = Spectrum(0.)
			if wi_w.dot(self.ng) * wo_w.dot(self.ng) > 0.:
				# ignore BTDF
				flags = BDFType(flags & ~BDFType.TRANSMISSION)
			else:
				# ignore BRDF
				flags = BDFType(flags & ~BDFType.REFLECTION)

			for func in self.bdfs:
				if func.match_flag(flags):
					f += func.f(wo, wi)

		return [pdf, wi_w, bdf.type, f]

	def pdf(self, wo: 'Vector', wi: 'Vector', flags: 'BDFType'=BDFType.ALL) -> FLOAT:
		if self.n_BDF == 0: return 0.
		wo = self.w2l(wo_w)
		wi = self.w2l(wi_w)
		pdf = 0.
		n_match = 0
		for bdf in self.bdfs:
			if bdf.match_flag(flags):
				n_match += 1
				pdf += bdf.pdf(wo, wi)

		return pdf / n_match if n_match > 0 else 0.

	def push_back(self, bdf: 'BDF'):
		'''
		Add more BDF
		'''
		self.bdfs.append(bdf)
		self.__nBDF += 1

	@property
	def n_BDF(self):
		return self.__nBDF

	def n_components(self, flags: 'BDFType'):
		n = 0
		for bdf in self.bdfs:
			if bdf.match_flag(flags):
				n += 1
		return n

	def w2l(self, v: 'Vector') -> 'Vector':
		'''
		w2l()

		Transform a `Vector` in world in the local
		surface normal coord. system
		'''
		return Vector(v.dot(self.sn), v.dot(self.tn), v.dot(self.nn))

	def l2w(self, v: 'Vector') -> 'Vector':
		'''
		l2w()

		Transform a `Vector` in the local system
		to the world system
		'''
		return Vector(  self.sn.x * v.x + self.tn.x * v.x + self.nn.x * v.x,
						self.sn.y * v.y + self.tn.y * v.y + self.nn.y * v.y,
						self.sn.z * v.z + self.tn.z * v.z + self.nn.z * v.z )

	@jit
	def f(self, wo_w: 'Vector', wi_w: 'Vector', flags: 'BDFType') -> 'Spectrum':
		wi = self.w2l(wi)
		wo = self.w2l(wo)	

		if (wi_o.dot(ng)) * (wo_w.dot(ng)) < 0.:
			# no transmission
			flags = BDFType(flags & ~BDFType.TRANSMISSION)
		else:
			# no reflection
			flags = BDFType(flags & ~BDFType.REFLECTION)

		f = Spectrum(0.)

		for bdf in self.bdfs:
			if bdf.match_flag(flags):
				f+= bdf.f(wo, wi)

		return f

	def rho_hd(self, wo: 'Vector', flags: 'BDFType'=BDFType.ALL, sqrt_samples: INT=6, rng=np.random.rand) -> 'Spectrum':
		'''
		Computs hemispherical-directional reflectance function.

		'''
		nSamples = sqrt_samples * sqrt_samples
		smp = stratified_sample_2d(sqrt_samples, sqrt_samples, rng=rng)

		sp = Spectrum(0.)
		for bdf in self.bdfs:
			if bdf.match_flag(flags):
				sp += bdf.rho_hd(wo, smp)

		return sp

	def rho_hh(self, flags: 'BDFType'=BDFType.ALL, sqrt_samples: INT=6, rng=np.random.rand) -> 'Spectrum':
		'''
		Computs hemispherical-hemispherical reflectance function.

		'''
		nSamples = sqrt_samples * sqrt_samples
		smp_1 = stratified_sample_2d(sqrt_samples, sqrt_samples, rng=rng)
		smp_2 = stratified_sample_2d(sqrt_samples, sqrt_samples, rng=rng)

		sp = Spectrum(0.)
		for bdf in self.bdfs:
			if bdf.match_flag(flags):
				sp += bdf.rho_hh(nSamples, smp_1, smp_2)

		return sp









	



