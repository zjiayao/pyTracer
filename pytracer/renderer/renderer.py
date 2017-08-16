"""
renderer.py

Renderer Class

Created by Jiayao on Aug 5, 2017
"""
from __future__ import absolute_import
from abc import (ABCMeta, abstractmethod)
from pytracer import *
import pytracer.geometry as geo

__all__ = ['Renderer', 'SamplerRenderer', 'SamplerRendererTask']


class Renderer(object, metaclass=ABCMeta):
	"""
	Renderer Class
	"""

	@abstractmethod
	def render(self, scene: 'Scene'):
		raise NotImplementedError('src.core.renderer.{}.render(): abstract method '
							'called'.format(self.__class__)) 

	@abstractmethod
	def li(self, scene: 'Scene', ray: 'geo.RayDifferential', sample: 'Sample',
			rng='np.random.rand') -> ['Spectrum', 'Intersection']:
		raise NotImplementedError('src.core.renderer.{}.li(): abstract method '
							'called'.format(self.__class__)) 

	@abstractmethod
	def transmittance(self, scene: 'Scene', ray: 'geo.RayDifferential', sample: 'Sample',
									rng='np.random.rand') -> 'Spectrum':
		raise NotImplementedError('src.core.renderer.{}.transmittance(): abstract method '
							'called'.format(self.__class__)) 


class SamplerRenderer(Renderer):
	"""
	SamplerRenderer Class

	Sample-driven renderer
	"""
	def __init__(self, s: 'Sampler', c: 'Camera', si: 'SurfaceIntegrator', vi: 'VolumeIntegrator'):
		self.sampler = s
		self.camera = c
		self.surf_integrator = si
		self.vol_integrator = vi

	def li(self, scene: 'Scene', ray: 'geo.RayDifferential', sample: 'Sample',
			rng=np.random.rand) -> ['Spectrum', 'Intersection']:
		# local variables
		assert ray.time == sample.time

		li = Spectrum(0.)
		is_hit, isect = scene.intersect(ray)
		if is_hit:
			li = self.surf_integrator.li(scene, self, ray, isect, sample, rng)
		else:
			for light in scene.lights:
				li += light.le(ray)

		if self.vol_integrator is not None:
			lvi, T = self.vol_integrator.li(scene, self, ray, sample, rng)

			return [T * li + lvi, T, isect]
		else:
			return [li, Spectrum(0.), isect]

	def transmittance(self, scene: 'Scene', ray: 'geo.RayDifferential', sample: 'Sample',
									rng='np.random.rand') -> 'Spectrum':
		return self.vol_integrator.transmittance(scene, self, ray, sample, rng)

	def render(self, scene: 'Scene'):
		from pytracer.sampler import Sample
		# integrator proprocessing
		if self.surf_integrator is not None:
			self.surf_integrator.preprocess(scene, self.camera, self)
		if self.vol_integrator is not None:
			self.vol_integrator.preprocess(scene, self.camera, self)

		# init sample
		sample = Sample(self.sampler, self.surf_integrator, self.vol_integrator,scene)

		task = SamplerRendererTask(scene, self, self.camera, self.sampler, sample, False, 1, 1)
		task()		
		# main rendering loop: launch tasks
		# numer of samples
		# n_pixels = self.camera.film.xResolution * self.camera.film.yResolution
		# n_tasks = max(32 *  multiprocessing.cpu_count(), n_pixels / (16 * 16))
		# n_tasks = round_pow_2(n_tasks)
		# n_tasks = 1
		# render_tasks = []
		# for i in range(n_tasks):
		#   render_tasks.append(SamplerRendererTask(scene, self, self.camera,
		#       self.sampler, sample, n_tasks-1-i, n_tasks))
		# enqueue_tasks(render_tasks)
		# wait_for_tasks()
		# render_tasks[0]()
		# store result
		self.camera.film.write_image()


class SamplerRendererTask(object):
	"""
	SamplerRendererTask Class
	"""
	def __init__(self, sc: 'Scene', ren: 'Renderer', c: 'Camera',
					ms: 'Sampler', smp: 'Sample', vis_obj_id: bool, tn: INT, tc: INT):
		self.scene = sc
		self.renderer = ren
		self.camera = c
		self.main_sampler = ms
		self.orig_sample = smp
		self.task_num = tn
		self.task_cnt = tc
		self.vis_obj_id = vis_obj_id

	def __repr__(self):
		return "{}\n{}/{}".format(self.__class__, self.task_num, self.task_cnt)

	def __call__(self):
		# get sub-sampler
		# sampler = self.main_sampler.get_subsampler(self.task_num, self.task_cnt)
		sampler = self.main_sampler
		if sampler is None:
			return

		# variables for rendering loop
		# memory managed by python
		rng = np.random.rand

		# allocate space for samples and isects
		max_smp = sampler.maximum_sample_cnt()
		rays = [None] * max_smp
		Ls = [None] * max_smp
		Ts = [None] * max_smp
		isects = [None] * max_smp

		# get samples and update image
		total_iteration = (sampler.xPixel_end - sampler.xPixel_start) * (sampler.yPixel_end - sampler.yPixel_start)
		for cnt, samples in enumerate(sampler):
			assert(samples is not None)

			util.progress_reporter(cnt, total_iteration)

			# generate camera ray and compute radiance
			for i, sample in enumerate(samples):
				# print('    Sample: {}/{}\n'.format(i+1, cnt))

				# find camera ray for i-th sample
				wt, rays[i] = self.camera.generate_ray_differential(sample)
				rays[i].scale_differential(1. / np.sqrt(sampler.spp))

				hit, isects[i] = self.scene.intersect(rays[i])

				if self.vis_obj_id:
					if hit and wt > 0.:
						# random shading
						Ls[i] = Spectrum.from_rgb([.5, .6, .7])
					else:
						Ls[i] = Spectrum(0.)
				else:
					if wt > 0.:
						Ls[i], Ts[i], isects[i] = self.renderer.li(self.scene,rays[i],sample, rng)
						Ls[i] *= wt

					else:
						Ls[i] = Spectrum(0.)
						Ts[i] = Spectrum(1.)

				if Ls[i].has_nans():
					util.logging('Error', 'NAN radiance returned, setting to black.')
					Ls[i] = Spectrum(0.)
				elif Ls[i].y() < -EPS:
					util.logging('Error', 'Negative luminance {} returned, setting to black.'.format(Ls[i].y()))
					Ls[i] = Spectrum(0.)
				elif Ls[i].y() == np.inf:
					util.logging('Error', 'Infinite luminance returned, setting to balck.')
					Ls[i] = Spectrum(0.)

			# report results, add contribution
			if sampler.report_results(samples, rays, Ls, isects):
				for i, sample in enumerate(samples):
					self.camera.film.add_sample(sample, Ls[i])













