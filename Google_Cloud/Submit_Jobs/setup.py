'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='trainer',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='MNIST MLP keras model on Cloud ML Engine',
      author='Maggie Cao',
      author_email='mahgieeee@hotmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py',
          'pillow',
          'joblib',
          'opencv-python'],
      zip_safe=False)
