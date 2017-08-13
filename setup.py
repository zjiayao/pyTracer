from distutils.core import setup

setup(
    name='pyTracer',
    version='0.1dev',
	description='Python-based Photorealistic Ray Tracer',
	author='Jiayao Zhang',
	author_email='jyzhang@cs.hku.hk',
	url='https://i.cs.hku.hk/~jyzhang/',
    packages=['pytracer'],
    license='MIT License',
    long_description=open('Readme.md').read(),
)