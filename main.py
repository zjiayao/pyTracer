'''
main.py

Entrance point for PyTracer

Created by Jiayao on Aug 5, 2017
'''
from numba import jit
import numpy as np
from scipy import spatial
import PIL

import src.core.pytracer
import src.core.backend as pytracer

def main():
	# process cmd args
	filename == []
	# init
	pytracer.init()
	if len(filename) == 0:
		# from cmd input
		pytracer.parse('-')
	else:
		# from file
		for file in filename:
			try:
				pytracer.parse(file)
			except:
				print("PyTracer.main: Cannot parse file {}, abort".format(file))
				


if __name__ == '__main__':
	main()