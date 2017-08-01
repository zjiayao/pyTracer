'''
pytracer.py

This module is part of the pyTracer, which
holds global definitions

Created by Jiayao on July 27, 2017
'''

from __future__ import print_function
import os, sys, glob

from numba import jit
import numpy as np
import PIL.Image

EPS = 1e-5
INT = int
UINT = np.uint32
FLOAT = np.float64
DOUBLE = np.float64
HANDNESS = 'left'

INV_2PI = 1. / (2. * np.pi)

FILTER_TABLE_SIZE = 16

# Global Functions

def feq(x: FLOAT, y: FLOAT):
	return np.isclose(x, y, atol=EPS)

def eq_unity(x: FLOAT):
	return x > 1. - EPS and x < 1. + EPS	

def ne_unity(x: FLOAT):
	return x < 1. - EPS or x > 1. + EPS	

def Lerp(t: FLOAT, v1: FLOAT, v2: FLOAT):
	'''
	Lerp
	Linear interpolation between `v1` and `v2`
	'''
	return (1. - t) * v1 + t * v2

ufunc_lerp = np.frompyfunc(Lerp, 3, 1)

@jit
def write_image(filename: str, rgb: 'np.ndarray', alpha: 'np.ndarray',
				xRes: INT, yRes: INT, xFullRes: INT, yFullRes: INT,
				xOffset: INT, yOffset: INT):

	if filename is None or rgb is None:
		raise RuntimeError('scr.core.pytracer.write_image(): filename and rgb cannot '
							'be None')
	for i in range(3):
		rgb[:,:,i] = (255 * rgb[:,:,i]) / np.max(rgb[:,:,i])
	
	rgb.astype(np.uint8)

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