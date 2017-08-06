'''
renderer.py

Renderer Class

Created by Jiayao on Aug 5, 2017
'''
from numba import jit
import numpy as np
from abc import ABCMeta, abstractmethod

import  multiprocessing

from src.core.pytracer import *
from src.core.scene import *

class Renderer(object, metaclass=ABCMeta):
	'''
	Renderer Class
	'''

	@abstractmethod
	def render(self, scene: 'Scene'):
		raise NotImplementedError('src.core.renderer.{}.render(): abstract method '
							'called'.format(self.__class__)) 

	@abstractmethod
	def li(self, scene: 'Scene', ray: 'RayDifferential', sample: 'Sample',
			rng='np.random.rand', isect: 'Intersection'=None) -> 'Spectrum':
		raise NotImplementedError('src.core.renderer.{}.li(): abstract method '
							'called'.format(self.__class__)) 


	@abstractmethod
	def transmittance(self, scene: 'Scene', ray: 'RayDifferential', sample: 'Sample',
									rng='np.random.rand') -> 'Spectrum':
		raise NotImplementedError('src.core.renderer.{}.transmittance(): abstract method '
							'called'.format(self.__class__)) 


class SamplerRenderer(Renderer):
	'''
	SamplerRenderer Class

	Sample-driven renderer
	'''
	def __init__(self, s: 'Sampler', c: 'Camera', si: 'SurfaceIntegrator', vi: 'VolumeIntegrator'):
		self.sampler = s
		self.camera = c
		self.surf_integrator = si
		self.vol_integrator = vi

	def li(self, scene: 'Scene', ray: 'RayDifferential', sample: 'Sample',
			rng='np.random.rand', isect: 'Intersection'=None) -> 'Spectrum':
		# local variables 
		if isect is None:
			isect = Intersection()

		li = Spectrum(0.)
		is_hit, isect = scene.intersect(ray)
		if is_hit:
			li = self.surf_integrator.li(scene, self, ray, isect, sample, rng)
		else:
			for light in scene.lights:
				li += light.le(ray)


		lvi, T = self.vol_integrator.li(scene, self, ray, sample, rng)

		return T * li + lvi

	def transmittance(self, scene: 'Scene', ray: 'RayDifferential', sample: 'Sample',
									rng='np.random.rand') -> 'Spectrum':
		return self.vol_integrator.transmittance(scene, self, ray, sample, rng)

	def render(self, scene: 'Scene'):
		# integrator proprocessing
		self.surf_integrator.preprocess(scene, self.camera, self)
		self.vol_integrator.preprocess(scene, self.camera, self)

		# init sample
		sample = Sample(self.sampler, self.surf_integrator, self.vol_integrator,
							scene)		
		
		# main rendering loop: launch tasks
		## numer of samples
		n_pixels = self.camera.film.xResolution * self.camera.film.yResolution
		n_tasks = max(32 *  multiprocessing.cpu_count(), n_pixels / (16 * 16))
		n_tasks = round_pow_2(n_tasks)
		
		render_tasks = []
		for i in range(n_tasks):
			render_tasks.append(SamplerRendererTask(scene, self, self.camera, 
									self.sampler, sample, n_tasks-1-i, n_tasks))
		enqueue_tasks(render_tasks)
		wait_for_tasks()

		# store result
		self.camera.film.write_image()



class SamplerRendererTask():
	'''
	SamplerRendererTask Class
	'''
	def __init__(self, sc: 'Scene', ren: 'Renderer', c: 'Camera',
					ms: 'Sampler', smp: 'Sample', tn: INT, tc: INT):
		self.scene = sc
		self.renderer = ren
		self.camera = c
		self.main_sampler = ms
		self.orig_sample = smp
		self.task_num = tn
		self.task_cnt = tc

	def __repr__(self):
		return "{}\n{}/{}".format(self.__class__, self.task_num, self.task_cnt)

	def __call__(self):
		# get sub-sampler
		sampler = self.main_sampler.get_subsampler(self.task_num, self.task_cnt)
		if sampler is None:
			return

		# variables for rendering loop
		# memory managed by pytohn
		rng = np.random.rand

		# allocate space for samples and isects
		max_smp = sampler.maximum_sample_cnt()
		rays = [RayDifferential() for _ in range(max_smp)]
		Ls = [Spectrum() for _ in range(max_smp)]
		Ts = [Spectrum() for _ in range(max_smp)]
		isects = [Intersection() for _ in range(max_smp)]

		# get samples and update image
		for samples in sampler:
			assert(samples is not None)
			cnt = len(samples)
			# generate camera ray and compute radiance
			for i, sample in enumerate(samples):
				## find camera ray for i-th sample
				wt, rays[i] = self.camera.generate_ray_differential(sample)
				rays[i].scale_differential(1. / np.sqrt(sampler.spp))

				## evaluate radiance
				if wt > 0.:
					Ls[i], Ts[i] = wt * self.renderer.li(self.scene, ray, sample, rng, isects[i])
				else:
					Ls[i] = Spectrum(0.)
					Ts[i] = Spectrum(1.)
			# report results, add contribution
			if sampler.report_results(samples, ray, Ls, isect):
				for i, sample in enumerate(samples):
					self.camera.film.add_sample(sample, ls[i])













