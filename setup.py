from distutils.core import setup, Extension


#c++11 standard libraries required

#Path to Minuit library
minuitpath="/home/farmer/apps/Minuit-1_7_9/src/.libs"

simulator = Extension('clstools.simulator',
                      sources = ['clstools/simulator.cpp'],
                      include_dirs =['clstools'],
                      extra_compile_args=["-std=c++11"],
                      library_dirs = [minuitpath],
                      libraries = ['stdc++','lcg_Minuit'],
                     )
setup(
    name='clstools',
    version='0.1.0',
    author='Benjamin Farmer',
    author_email='ben.farmer@gmail.com',
    packages=['clstools'],
    ext_modules = [simulator],
    scripts=[],
    url='http://pypi.python.org/pypi/hdf5tools/', #not really, just an example
    license='LICENSE.txt',
    description='A collection of tools for setting CLs limits, primarily on models predicting the results of Poisson counting experiments',
    long_description=open('README.txt').read(),
     #install_requires=[
    #    "Minuit v1.7.9",
    #    "PyMC",
    #    "numpy",
    #    "scipy",
    #    "matplotlib",
    #],
)
