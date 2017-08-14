"""
image.py


pytracer.texture.texture package

Image Texture

Created by Jiayao on Aug 5, 2017
Modified on Aug 14, 2017
"""
from __future__ import absolute_import
from pytracer import *
import pytracer.geometry as geo
import pytracer.spectral as spec
import pytracer.utility.imageio as iio
from pytracer.texture.texture import Texture
from pytracer.texture import MIPMap


class ImageTexture(Texture):
	"""
	ImageTexture Class

	Types used in memory storage and returning
	may differ. Thus two type parameters must be passed
	at initiation, i.e., `type_ret` the returning type
	when evaluating the `Texture` and `type_mem` the type
	used to storing texture in memory.
	"""

	class TexInfo(object):
		"""
		Wrapper class for
		keying different image
		texures.
		"""

		def __init__(self, filename: str, trilinear: bool, max_aniso: FLOAT, wrap: 'ImageWrap',
		             scale: FLOAT, gamma: FLOAT):
			self.filename = filename
			self.trilinear = trilinear
			self.max_aniso = max_aniso
			self.wrap = wrap
			self.scale = scale
			self.gamma = gamma

		def __repr__(self):
			return "{}\nFilename: {}\nTrilinear: {}\nMax Aniso: {}, Wrap: {}\nScale: {}, Gamma: {}" \
				.format(self.__class__, self.trilinear, self.max_aniso, self.wrap, self.scale,
			            self.gamma)

		def __hash__(self):
			return hash((self.filename, self.trilinear, self.max_aniso, self.wrap,
			             self.scale, self.gamma))

		def __eq__(self, other):
			return (self.filename == other.filename) and \
			       (self.trilinear == other.trilinear) and \
			       (self.max_aniso == other.max_aniso) and \
			       (self.wrap == other.wrap) and \
			       (self.scale == other.scale) and \
			       (self.gamma == other.gamma)

		def __ne__(self, other):
			return not (self == other)

	# class static
	textures = {}

	def __init__(self, type_ret: type, type_mem: type, mapping: 'TextureMapping2D', filename: str, trilinear: bool,
	             max_aniso: FLOAT, wrap: 'ImageWrap', scale: FLOAT, gamma: FLOAT):
		self.type_ret = type_ret
		self.type_mem = type_mem

		self.mapping = mapping
		self.mipmap = ImageTexture.get_texture(filename, trilinear, max_aniso,
		                                       wrap, scale, gamma)

	def __call__(self, dg: 'geo.DifferentialGeometry'):
		mem = self.mipmap.look_up(self.mapping(dg))
		return ImageTexture.__convert__out__(mem, self.type_ret)

	@staticmethod
	def __convert__in__(texel: object, type_mem: type, scale: FLOAT, gamma: FLOAT):
		"""
		Convert input texel to storage type.

		Performs gamma correction if needed.
		"""

		if isinstance(texel, 'RGBSpectrum') and \
						type_mem is spec.RGBSpectrum:
			return (scale * texel) ** gamma
		elif isinstance(texel, 'RGBSpectrum') and \
						type_mem is (FLOAT or float):
			return (scale * texel.y()) ** gamma
		else:
			raise TypeError('src.core.texture.ImageTexture.__convert__in__: '
			                'undefined convertion between {} and {}'.format(texel.__class__, type_mem))

	@staticmethod
	def __convert__out__(texel: object, type_ret: type):
		"""
		Convert looking-up texture value
		from storage type to returning type.
		"""

		if isinstance(texel, 'RGBSpectrum') and \
						type_ret is Spectrum:
			return Spectrum.fromRGB(texel.toRGB())
		elif (isinstance(texel, float) or isinstance(texel, FLOAT)) and \
						type_ret is (float or FLOAT):
			return texel
		else:
			raise TypeError('src.core.texture.ImageTexture.__convert__in__: '
			                'undefined convertion between {} and {}'.format(texel.__class__, type_ret))

	@staticmethod
	def __clear__cache__():
		ImageTexture.textures = {}

	def get_texture(self, filename: str, trilinear: bool, max_aniso: FLOAT,
	                wrap: 'ImageWrap', scale: FLOAT, gamma: FLOAT) -> 'MIPMap':
		# look in the cache
		tex_info = ImageTexture.TexInfo(filename, trilinear, max_aniso, wrap, scale, gamma)
		if tex_info in ImageTexture.textures:
			return ImageTexture.textures[tex_info]

		try:
			texels = iio.read_image(filename)
			width, height = np.shape(texels)
		except:
			print('src.core.texture.{}.get_texture(): cannot process file {}, '
			      'use default one-valued MIPMap'.format(self.__class__, filename))
			texels = None

		if texels is not None:
			if not isinstance(texels, self.type_mem):
				conv = np.empty([height, width], dtype=object)
				for t in range(height):
					for s in range(width):
						conv[t, s] = ImageTexture.__convert_in__(texels[t, s], self.type_mem, scale, gamma)

			ret = MIPMap(self.type_mem, conv, trilinear, max_aniso, wrap)

		else:
			# one-valued MIPMap
			val = [[self.power(scale, gamma)]]
			ret = MIPMap(self.type_mem, val)

		ImageTexture.textures[tex_info] = ret
		return ret


