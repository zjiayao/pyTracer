# PyTracer-master

[![Build Status](https://travis-ci.com/zjiayao/pyTracer.svg?token=9cK4Kmeqpdioyfb1EXxS&branch=master)](https://travis-ci.com/zjiayao/pyTracer)
[![Chat on Gitter](https://badges.gitter.im/zjiayao/pyTracer.svg)](https://gitter.im/zjiayao/pyTracer/)


## Introduction

This is the developement (`master`) branch of PyTracer. For more infomration, see `stable` branch.


## Development

This branch is now focusing on `Cython`-izing the PyTracer in hope for
better performance.  Currently, the following components are `Cython`-ized:

- `geometry/`
- `transform/`, without `quaternion`
- `spectrum/`
- `shape/shape.py`


## Cite This Project


    @misc{pytracer:2017,
		title = {pyTracer},
		year = {2017},
		author = {Jiayao, Zhang and Li-Yi, Wei},
		publisher = {GitHub},
		journal = {GitHub Repository},
		howpublished= {\url{https://github.com/zjiayao/pyTracer}
    }
