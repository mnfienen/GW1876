#from distutils.core import setup

from setuptools import setup
'''
try:
    from pypandoc import convert
    
    read_md = lambda f: convert(f, 'rst')
    long_descrip = read_md('README.md')
except ImportError:
    long_descrip = open('README.md').readlines()
'''
long_descrip = ''
def run():
    setup(name="PestTools",
        version="0.0.1",
        description="Python package for processing and visualizing information associated with the parameter estimation software PEST",
        long_description=long_descrip,
        url="",
        license="MIT",
        author="Evan Christianson, Andrew Leaf, Jeremy White, Mike Fienen",
        maintainer_email="",
        packages=["pestools"])

if __name__ == "__main__":
    run()


