from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

src = ['definition.pxd']

extensions = [
	Extension("pytracer.core.definition", ["pytracer/core/definition.pyx"],
	          language='c++'),
	Extension("pytracer.geometry.geometry", ["pytracer/geometry/geometry.pyx"],
                        language='c++')
]

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