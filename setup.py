from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

src = ['pytracer.geometry.geometry']
loc = [['pytracer/geometry/geometry.pyx']]
incl = [numpy.get_include(), 'pytracer/include']


def _construct_ext(source: list, location: list, include: list):
	exts = []
	for i, modu in enumerate(source):
		exts.append(Extension(modu, location[i], include_dirs=include, language='c++'))
	return exts

extensions = _construct_ext(src, loc, incl)

# extensions = [
# 	Extension("pytracer.core.definition", ["pytracer/core/definition.pyx"],
# 	          language='c++'),
# 	Extension("pytracer.geometry.geometry", ["pytracer/geometry/geometry.pyx"],
#                         language='c++')
# ]

setup(
	# name='pyTracer',
	# version='0.1dev',
	# description='Python-based Photorealistic Ray Tracer',
	# author='Jiayao J. Zhang and Li-Yi Wei',
	# author_email='{jyzhang, lywei}@cs.hku.hk',
	# url='https://zjiayao.github.com/pytracer/',
	# packages=['pytracer'],
	# license='MIT License
	# long_description=open('Readme.md').read(),
	ext_modules=cythonize(extensions)
	# ext_modules=cythonize("pytracer/geometry/geometry.pyx",
	#                       include_path=['./pytracer/core', '.'], language='c++')#cythonize("pytracer/*.pyx", language='c++', include_path=['.', './pytracer/core']),
)