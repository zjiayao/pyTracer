"""
interface.py

pytracer.interface package

Interfacing from the scene descriptions.


Created by Jiayao on Aug 20, 2017
"""
from __future__ import absolute_import
from pytracer.interface.parameter import *
from pytracer.interface.option import *
from pytracer.interface.api import *

# State Variables
# API Initialization
API_UNINIT = 0
API_OPTIONS = 1
API_WORLD = 2
API_STATUS = API_UNINIT

# Option
GLOBAL_OPTION = None
RENDER_OPTION = None

GRAPHICS_STATE = None
GRAPHICS_STATE_STACK = []
TRANSFORM_STACK = []

# Transforms
MAX_TRANSFORM = 2
TRANSFORM_SET = [None] * MAX_TRANSFORM
COORDINATE_SYSTEM = {}