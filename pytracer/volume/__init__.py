"""
__init__.py


pytracer.texture.texture package

Texture interface and basic textures.

Generic interface.
NB: Instanciating types need supporting
copy() method.

Created by Jiayao on Aug 5, 2017
Modified on Aug 14, 2017
"""
from __future__ import absolute_import
from pytracer.volume.utility import *
from pytracer.volume.bssrdf import *
from pytracer.volume.volume import *

__all__ = ['subsurface_from_diffuse', 'BSSRDF',
           'VolumeRegion', 'DensityRegion']