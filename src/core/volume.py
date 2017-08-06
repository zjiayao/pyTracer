'''
volume.py

Model volume scatterings.

Created by Jiayao on Aug 7, 2017
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


# Global Utility Functions
## Phase Functions
@jit
def phase_isotrophic(w: 'Vector', wp: 'Vector') -> FLOAT:
	return 1. / (4. * PI

@jit
def phase_rayleigh(w: 'Vector', wp: 'Vector') -> FLOAT:
	ct = w.dot(wp)
	return 3. / (16. * PI) * (1. + ct * ct)

@jit
def phase_mie_hazy(w: 'Vector', wp: 'Vector') -> FLOAT:
	ct = w.dot(wp)
	return (.5 + 4.5 * np.power(.5 * (1. + ct), 8.)) / (4. * PI)
	
@jit
def phase_mie_murky(w: 'Vector', wp: 'Vector') -> FLOAT:
	ct = w.dot(wp)
	return (.5 + 16.5 * np.power(0.5 * (1. + ct), 32.)) / (4. * PI);

@jit
def phase_hg(w: 'Vector', wp: 'Vector', g: FLOAT) -> FLOAT:
	ct = w.dot(wp)
	return ((1. - g * g) / np.power(1. + g * g - 2. * g * ct, 1.5)) / (4. * PI);

@jit
def phase_schlick(w: 'Vector', wp: 'Vector', g: FLOAT) -> FLOAT:
	alpha = 1.5
	k = alpha * g + (1. - alpha) * g * g * g
	kct = k * w.dot(wp)
	return ((1. - k * k) / (1. - kct) * (1. - kct)) / (4. * PI);

	