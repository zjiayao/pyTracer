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
from scipy import spatial # cKdTree
import PIL.Image

# Type Alias

KdTree = spatial.cKDTree

# Global Constant

EPS = 1e-5
INT = int
UINT = np.uint32
FLOAT = np.float64
DOUBLE = np.float64
HANDNESS = 'left'

PI = FLOAT(np.pi)
INV_PI = 1. / np.pi
INV_2PI = 1. / (2. * np.pi)

FILTER_TABLE_SIZE = 16

# Global Static
IrIsotropicData = {}
ReHalfangleData = {}

# Global Functions

def feq(x: FLOAT, y: FLOAT):
	return np.isclose(x, y, atol=EPS)

def eq_unity(x: FLOAT):
	return x > 1. - EPS and x < 1. + EPS	

def ne_unity(x: FLOAT):
	return x < 1. - EPS or x > 1. + EPS	

def ftoi(x: FLOAT): return INT(np.floor(x))
def ctoi(x: FLOAT): return INT(np.ceil(x))
def rtoi(x: FLOAT): return INT(np.round(x))

@jit
def Lerp(t: FLOAT, v1: FLOAT, v2: FLOAT):
	'''
	Lerp
	Linear interpolation between `v1` and `v2`
	'''
	return (1. - t) * v1 + t * v2

def round_pow_2(x: INT) -> UINT: return UINT(2 ** np.round(np.log2(x)))
def next_pow_2(x: INT) -> UINT: return UINT(2 ** np.ceil(np.log2(x)))
def is_pow_2(x: INT) -> bool: return True if x == 0 else (np.log2(x) % 1) == 0.

ufunc_lerp = np.frompyfunc(Lerp, 3, 1)

# Utilities
from src.core.spectrum import *


def read_image(filename: str) -> 'Spectrum':
	'''
	Reads an image and returns a list
	of `Spectrum` list, width and height
	'''
	if filename is None:
		raise IOError('scr.core.pytracer.read_image(): filename cannot be None')

	specs = None
	with PIL.Image.open(filename) as pic:
		img = np.array(pic)
		width, height, chn = np.shape(img) # note the order of height and width

		# avoid expensive resizing in src.core.texture.MIPMap.__init__()
		if (not is_pow_2(width)) or (not is_pow_2(height)):
			# resample to power of 2
			width = next_pow_2(sres)
			height = next_pow_2(tres)
			pic.resize((width, height))
			img = np.array(pic)

		specs = np.empty([height, width], dtype=object)
		if chn == 3:
			for t in range(height):
				for s in range(width):
					specs[t, s] = Spectrum.fromRGB(img[t,s,:])
		# monochrome
		elif chn == 1:
			for t in range(height):
				for s in range(width):
					specs[t, s] = img[t,s]	

		else:
			raise TypeError('src.core.pytracer.read_image(): unsupported '
				'type of file {}'.format(filename))

	return specs


@jit
def write_image(filename: str, rgb: 'np.ndarray', alpha: 'np.ndarray',
				xRes: INT, yRes: INT, xFullRes: INT, yFullRes: INT,
				xOffset: INT, yOffset: INT):

	if filename is None or rgb is None:
		raise IOError('scr.core.pytracer.write_image(): filename and rgb cannot '
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