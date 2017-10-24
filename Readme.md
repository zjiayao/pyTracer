# pyTracer
![Head Model](examples/head.png)

*Proudly rendered with `pytracer`, head model courtesy of CS238b.*

[![Build Status](https://travis-ci.com/zjiayao/pyTracer.svg?token=9cK4Kmeqpdioyfb1EXxS&branch=stable)](https://travis-ci.com/zjiayao/pyTracer)
[![Chat on Gitter](https://badges.gitter.im/zjiayao/pyTracer.svg)](https://gitter.im/zjiayao/pyTracer/)

## Introduction

`pyTracer` can only do one thing, but it does it very well, albeit to
*a little bit* slow: photorealistic ray tracing.
The development of `pyTracer` largely takes references from *Physically Based Rendering*,
both second and thrid edition.

## Working Flow

The normal working flow is essentially the same
as the main rendering loop specified in `pbrt`;
A concrete example is given in `head.py`.

## Features

![Matte Example](examples/matte.png)

Currently, `pyTracer` supports the following
features:

- Triangle Mesh with Loop Subdividing Surface Modeling
- Full Spectrum Rendering (Sampling Spectrums Supported)
- Bounding Volume Hierarchy Accelerator
- Projective, Perspective and Orthographic Cameras
- Multiple Types of Textures
- Multiple Types of Materials including Irregular Sampled Isotropic BRDF
- Spot and Area Diffuse Lights
- Monte Carlo Integration with MCMC Samplers
- (Single) Path Integrator and Direct Lighting Integrater

Users familiar with `pbrt` may find it intuitive to work with other components.

## Development

Currently, `pyTracer` is still under development for supporting more features,
which are, tentatively:

- MERL BRDF Support
- Volume Scattering Modeling
- Bidirectional Path Tracing
- Direct support to `pbrt` flavour input files
- Optimization with `Cython`

## Cite This Project

    @misc{pytracer:2017,
		title = {pyTracer},
		year = {2017},
		author = {Jiayao, Zhang and Li-Yi, Wei},
		publisher = {GitHub},
		journal = {GitHub Repository},
		howpublished= {\url{https://github.com/zjiayao/pyTracer}
    }
