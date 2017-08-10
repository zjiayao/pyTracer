'''
integrator.py

Model integrators.

Created by Jiayao on Aug 9, 2017
'''

from numba import jit
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
from src.core.pytracer import *
from src.core.geometry import *
from src.core.diffgeom import *
from src.core.spectrum import *
from src.core.reflection import *
from src.core.texture import *
from src.core.montecarlo import *

# Utility Functions

@jit
def compute_light_sampling_cdf(scene: 'Scene') -> 'Distribution1D':
	'''
	compute_light_sampling_cdf()

	Creates a one dimensional
	distribution based on the power
	of all lights in the scene.
	'''
	n_lights = len(scene.lights)
	power = []
	for light in scene.lights:
		power.append(light.power(scene).y())

	return Distribution1D(power)

