'''
shape.y

The base class of shapes.

Created by Jiayao on July 27, 2017
'''

class Shape(object):
 	"""
 	Shape Class

 	Base class of shapes.
 	"""

 	def __init__(self, obj2wld, wld2obj, reverse_orientation):
 		super().__init__()
 		self.obj2wld = obj2wld
 		self.wld2obj = wld2obj
 		self.reverse_orientation = reverse_orientation
 		