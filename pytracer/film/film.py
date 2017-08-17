"""
film.py

The base class to model film.

Created by Jiayao on Aug 1, 2017
"""


from __future__ import absolute_import
import threading
from abc import (ABCMeta, abstractmethod)
from pytracer import *
import pytracer.utility.imageio as iio
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from pytracer.filter import Filter


__all__ = ['Film', 'ImageFilm']


class Film(object, metaclass=ABCMeta):
	"""
	Film Class
	"""

	def __init__(self, xr: INT, yr: INT):
		self.xResolution = xr
		self.yResolution = yr

	def __repr__(self):
		return "{}\nResolution: {} * {}".format(self.__class__,
						self.xResolution, self.yResolution)

	@abstractmethod
	def add_sample(self, sample: 'CameraSample', spectrum: 'Spectrum'):
		raise NotImplementedError('src.core.film {}.add_sample(): abstract method '
									'called'.format(self.__class__))

	@abstractmethod
	def splat(self, sample: 'CameraSample', spectrum: 'Spectrum'):
		"""
		Used to sum the contribution
		around a pixel sample
		"""
		raise NotImplementedError('src.core.film {}.add_sample(): abstract method '
									'called'.format(self.__class__))

	@abstractmethod
	def get_sample_extent(self) -> [INT]:
		"""
		Determine the range of pixels to generate
		samples, returns [xStart, xEnd, yStart, yEnd]
		"""
		raise NotImplementedError('src.core.film {}.get_sample_extent(): abstract method '
									'called'.format(self.__class__))

	@abstractmethod
	def get_pixel_extent(self) -> [INT]:
		"""
		Determine the range of pixels on
		the actual image, returns [xStart, xEnd, yStart, yEnd]
		"""
		raise NotImplementedError('src.core.film {}.get_pixel_extent(): abstract method '
									'called'.format(self.__class__))

	def update_display(self, x0: INT, y0: INT, x1: INT, y1: INT, splat_scale: FLOAT) -> [INT]:
		"""
		Update the window while rendering,
		if needed
		"""
		pass


	@abstractmethod
	def write_image(self, splat_scale: FLOAT=1.) -> [INT]:
		"""
		Display or write image to file
		"""
		raise NotImplementedError('src.core.film {}.write_image(): abstract method '
									'called'.format(self.__class__))


class ImageFilm(Film):
	"""
	ImageFilm Class

	Filter samples with a given reconstruction
	filter
	"""

	class Pixel(object):
		"""
		Pixel Class

		Inner class of `ImageFilm`
		"""
		def __init__(self):
			self.Lxyz = np.zeros(3, dtype=FLOAT)
			self.splatXYZ = np.zeros(3, dtype=FLOAT)
			self.weight_sum = 0.
			self.lock = threading.Lock() # for multi-threading

		def __repr__(self):
			return "{}\nLocked: {}".format(self.__class__, self.lock.locked())

	def __init__(self, xr: INT, yr: INT, filt: 'Filter', crop: [FLOAT], fn: str):
		super().__init__(xr, yr)
		self.crop = crop.copy()	# the extent of pixels to actually process
								# in NDC space, range: [0, 1]
		self.filename = fn
		self.filter = filt

		# compute film image extent
		self.xPixel_start = INT(np.ceil(xr * crop[0]))
		self.xPixel_cnt = np.maximum(1, INT(np.ceil(xr * crop[1])) - self.xPixel_start)
		self.yPixel_start = INT(np.ceil(xr * crop[2]))
		self.yPixel_cnt = np.maximum(1, INT(np.ceil(yr * crop[3])) - self.yPixel_start)

		# allocate storage
		self.pixels = np.empty([self.xPixel_cnt, self.yPixel_cnt], dtype=object)
		for i in range(self.xPixel_cnt):
			for j in range(self.yPixel_cnt):
				self.pixels[i][j] = self.Pixel()

		# precompute filter table
		# as an np array of np arrays
		self.filter_table = np.empty([FILTER_TABLE_SIZE, FILTER_TABLE_SIZE], dtype=FLOAT)
		dx = self.filter.xw / FILTER_TABLE_SIZE
		dy = self.filter.yw / FILTER_TABLE_SIZE
		for y in range(FILTER_TABLE_SIZE):
			fy = (y + .5) * dy
			for x in range(FILTER_TABLE_SIZE):
				self.filter_table[y, x] = self.filter((x + .5) * dx, fy)

	def add_sample(self, sample: 'CameraSample', L: 'Spectrum'):
		"""
		Assume different threads, if any,
		cannot mutate the same pixel at the same time
		"""
		# compute raster extent
		# (x0, x1) to (y0, y1)
		# inclusive

		dX = sample.imageX - .5
		dY = sample.imageY - .5
		x0 = util.ctoi(dX - self.filter.xw)
		x1 = util.ftoi(dX + self.filter.xw)
		y0 = util.ctoi(dY - self.filter.yw)
		y1 = util.ftoi(dY + self.filter.yw)

		x0 = np.maximum(x0, self.xPixel_start, dtype=INT)
		x1 = np.minimum(x1, self.xPixel_start + self.xPixel_cnt - 1, dtype=INT)
		y0 = np.maximum(y0, self.yPixel_start, dtype=INT)
		y1 = np.minimum(y1, self.yPixel_start + self.yPixel_cnt - 1, dtype=INT)

		if x1 - x0 < 0 or y1 - y0 < 0:
			return

		# add sample to pixel arrays
		from pytracer.spectral import rgb2xyz
		xyz = rgb2xyz(L)

		## precomputes offsets
		ifx = np.minimum(FILTER_TABLE_SIZE-1,
				np.floor(np.fabs((np.arange(x0, x1+1) - dX) * FILTER_TABLE_SIZE * self.filter.xwInv)).astype(INT) )
		ify = np.minimum(FILTER_TABLE_SIZE-1,
				np.floor(np.fabs((np.arange(y0, y1+1) - dY) * FILTER_TABLE_SIZE * self.filter.ywInv)).astype(INT) )		

		sync = self.filter.xw > .5 or self.filter.yw > .5
		### DEBUG
		### print("x0: ",  x0, " x1: ", x1, " y0: ", y0, " y1: ",y1)

		## synchronization needed if multiple pixels are covered
		for y in range(y0, y1+1):
			for x in range(x0, x1+1):
				# find filter value
				
				### DEBUG
				### print("x: ", type(x), " y: ", type(y), " x0: ", type(x0), " y0: ",type(y0))

				wt = self.filter_table[ify[y - y0]][ifx[x - x0]]
				pxl = self.pixels[x - self.xPixel_start][y - self.yPixel_start]

				if not sync:
					pxl.Lxyz += wt * xyz
					pxl.weight_sum += wt

				else:
					# note locks are stored in Pixel
					# which induce overhead
					# may use atomic ops later
					with pxl.lock:
					# pxl.lock.acquire()			# using `with` causes jit exception
						pxl.Lxyz += wt * xyz
						pxl.weight_sum += wt
					# pxl.lock.release()

	def splat(self, sample: 'CameraSample', L: 'Spectrum'):
		"""
		Used to sum the contribution
		around a pixel sample
		"""
		xyz = L.to_xyz()

		x = INT(np.floor(sample.imageX))
		y = INT(np.floor(sample.imageY))

		if x < self.xPixel_start or x - self.xPixel_start >= self.xPixel_cnt or \
				y < self.yPixel_start or y - self.yPixel_start >= self.yPixel_cnt:
			return

		pxl = self.pixels[x - self.xPixel_start][y - self.yPixel_start]
		pxl.lock.acquire()
		pxl.splatXYZ += xyz
		pxl.lock.release()

	def get_sample_extent(self) -> [INT]:
		"""
		Determine the range of pixels to generate
		samples, returns [xStart, xEnd, yStart, yEnd]
		"""
		return np.floor([self.xPixel_start + .5 - self.filter.xw,
						 self.xPixel_start + .5 + self.xPixel_cnt + self.filter.xw,
						 self.yPixel_start + .5 - self.filter.yw,
						 self.yPixel_start + .5 +self.yPixel_cnt + self.filter.yw]).astype(INT)

	def get_pixel_extent(self) -> [INT]:
		"""
		Determine the range of pixels on
		the actual image, returns [xStart, xEnd, yStart, yEnd]
		"""
		return [self.xPixel_start, self.xPixel_start + self.xPixel_cnt,
				self.yPixel_start, self.yPixel_start + self.yPixel_cnt]

	def write_image(self, splat_scale: FLOAT=1.) -> [INT]:
		"""
		Display or write image to file
		"""
		# convert to RGB and compute pixel values
		from pytracer.spectral import xyz2rgb
		rgb = np.empty([self.xPixel_cnt, self.yPixel_cnt, 3], dtype=FLOAT)

		for y in range(self.yPixel_cnt):
			for x in range(self.xPixel_cnt):
				
				rgb[y, x] = xyz2rgb(self.pixels[x][y].Lxyz)
				ws = self.pixels[x][y].weight_sum

				if not ws == 0.:
					rgb[y, x] = np.maximum(0., rgb[y, x] / ws)

				# add splat values
				rgb[y, x] += splat_scale * xyz2rgb(self.pixels[x][y].splatXYZ)

		# write image
		iio.write_image(self.filename, rgb, None,
		                self.xPixel_cnt, self.yPixel_cnt, self.xResolution, self.yResolution,
		                self.xPixel_start, self.yPixel_start)






