from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='pyTracer',
    version='0.1dev',
	description='Python-based Photorealistic Ray Tracer',
	author='Jiayao J. Zhang and Li-Yi Wei',
	author_email='{jyzhang, lywei}@cs.hku.hk',
	url='https://zjiayao.github.com/pytracer/',
	packages=['pytracer'],
	license='MIT License',
	long_description=open('Readme.md').read(),
	ext_modules=cythonize('.pyx'),
)