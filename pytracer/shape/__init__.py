"""
__init__.py

pytracer.shape package

Contains the base interface for shapes.

Implementation includes:
	- LoopSubdiv
	- TriangleMesh
	- Sphere
	- Cylinder
	- Disk

Created by Jiayao on July 27, 2017
Modified on Aug 13, 2017
"""
from __future__ import absolute_import

from pytracer.shape.shape import Shape
# from pytracer.shape.loopsubdiv import (create_loop_subdiv, LoopSubdiv)
# from pytracer.shape.triangle import *
# from pytracer.shape.sphere import *
# from pytracer.shape.cylinder import *
# from pytracer.shape.disk import *

__all__ =['Shape']
# __all__ = ['Shape', 'create_loop_subdiv','LoopSubdiv',
# 'create_triangle_mesh', 'TriangleMesh', 'Triangle',
# 'Sphere', 'Cylinder', 'Disk']

