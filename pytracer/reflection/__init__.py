"""
__init__.py

patracer.reflection package

Utility functions

Convention:
	Incident light and viewing direction
	are normalized and face outwards;
	Normal faces outwards and is not
	flipped to lie in the same side as
	viewing direction.

Created by Jiayao on Aug 13, 2017
"""
from __future__ import absolute_import
from pytracer.reflection.utility import *
from pytracer.reflection.bdf import *
from pytracer.reflection.fresnel import *
from pytracer.reflection.bsdf import *