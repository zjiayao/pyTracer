"""
imageio.py


Defines image IO utility
functions.

Created by Jiayao on July 27, 2017
Modfied on Aug 13, 2017
"""
import numpy as np
import PIL.Image
import os
from pytracer import INT
import pytracer.utility.utility as util
import pytracer.spectral as spec

__all__ = ['read_image', 'write_image']


def read_image(filename: str) -> ['spec.Spectrum']:
	"""
	Reads an image and returns a list
	of list of `Spectrum`s, width and height
	"""
	if filename is None:
		raise IOError('pytracer.imageio.read_image(): filename cannot be None')

	specs = None
	with PIL.Image.open(filename) as pic:
		img = np.array(pic)
		width, height, chn = np.shape(img) # note the order of height and width

		# avoid expensive resizing in src.core.texture.MIPMap.__init__()
		if (not util.is_pow_2(width)) or (not util.is_pow_2(height)):
			# re-sample to power of 2
			width = util.next_pow_2(width)
			height = util.next_pow_2(height)
			pic.resize((width, height))
			img = np.array(pic)
			return img
		# specs = np.empty([height, width], dtype=object)
		# if chn == 3:
		# 	for t in range(height):
		# 		for s in range(width):
		# 			specs[t, s] = spec.Spectrum.from_rgb(img[t, s, :])
		# # monochrome
		# elif chn == 1:
		# 	for t in range(height):
		# 		for s in range(width):
		# 			specs[t, s] = img[t,s]

		else:
			raise TypeError('pytracer.imageio.read_image(): unsupported '
				'type of file {}'.format(filename))


def write_image(filename: str, rgb: 'np.ndarray', alpha: 'np.ndarray',
                xRes: INT, yRes: INT, xFullRes: INT, yFullRes: INT,
                xOffset: INT, yOffset: INT):

	if filename is None or rgb is None:
		raise IOError('scr.core.pytracer.write_image(): filename and rgb cannot '
							'be None')
	for i in range(3):
		coef = np.max(rgb[:, :, i])
		if not coef == 0.:
			rgb[:, :, i] = (255 * rgb[:, :, i]) / coef
	
	rgb = rgb.astype('uint8')

	if alpha is None:
		if xRes == xFullRes and yRes == yFullRes:
			# save new img

			pic = PIL.Image.fromarray(rgb, mode="RGB")
			pic.save(filename, 'PNG')

		else:
			if not os.path.exists(filename):
				raise RuntimeError('scr.core.pytracer.write_image(): cannot find {} '
														'to update'.format(filename))
			# update img
			pic = PIL.Image.open(filename)
			img = np.array(pic)
			pic.close()

			img[xOffset:xOffset+xRes-1, yOffset:yOffset+yRes-1, :] = rgb
			
			pic = PIL.Image.fromarray(img, mode="RGB")
			pic.save(filename, 'PNG')

	else:
		# contains alpha channel
		raise NotImplementedError('src.core.pytracer.write_image(): coming soon')
