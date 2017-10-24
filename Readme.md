# PyTracer

[![Build Status](https://travis-ci.com/zjiayao/pyTracer.svg?token=9cK4Kmeqpdioyfb1EXxS&branch=stable)](https://travis-ci.com/zjiayao/pyTracer)
[![Chat on Gitter](https://badges.gitter.im/zjiayao/pyTracer.svg)](https://gitter.im/zjiayao/pyTracer/)

![Head Model](examples/head.png)

*Proudly rendered with `pytracer`, head model courtesy of CS238b.*


## Introduction


**PyTracer** is a photorealistic rendering interface
backended by ray tracing algorithm. It may sound
counter-intuitive that one may ever wish to
implement a ray tracer with `Python`, nevertheles,
PyTracer makes it feasible to do fast proto-typing
using current state-of-the-art learning libraries
which are mostly implemented in `Python`.

Indeed, albeit to being *a little bit* slow, **PyTracer**
can do it pretty well.
The development and implementation of **PyTracer**
largely take references from *Physically Based Rendering*,
both second and thrid edition. Its object-oriented nature
and modular design enables easy experiementing of new
algorithms.


## Working Flow


The normal working flow is essentially the same
as the main rendering loop specified in `pbrt`;
A concrete example is given in `head.py`,
try it by:

    >> import head


## Features


![Matte Example](examples/matte.png)


Currently, PyTracer supports the following
features:

- Triangle Mesh with Loop Subdividing Surface Modeling
- Full Spectrum Rendering (Sampling Spectrums Supported)
- Bounding Volume Hierarchy Accelerator
- Projective, Perspective and Orthographic Cameras
- Multiple Types of Textures
- Multiple Types of Materials including Irregular Sampled Isotropic BRDF
- Spot and Area Diffuse Lights
- Monte Carlo Integration with MCMC Samplers
- (Single) Path Integrator and Direct Lighting Integrator

Users familiar with `pbrt` may find it intuitive to work with other components.


## Development


PyTracer is still under development for supporting more features,
which are, tentatively:

- MERL BRDF Support
- Volume Scattering Modeling
- Bidirectional Path Tracing
- Direct support to `pbrt` flavour input files
- Optimization with `Cython`

And most importantly,

- **Making bullshits on performance real**


## Cite This Project


    @misc{pytracer:2017,
		title = {pyTracer},
		year = {2017},
		author = {Jiayao, Zhang and Li-Yi, Wei},
		publisher = {GitHub},
		journal = {GitHub Repository},
		howpublished= {\url{https://github.com/zjiayao/pyTracer}
    }
