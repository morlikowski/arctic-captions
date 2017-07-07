from setuptools import setup

setup(name='arctic-captions',
      version='0.1',
      description='An implementation of the Show, Tell and Attend neural captioning model.',
      url='https://github.com/morlikowski/arctic-captions',
      author='Matthias Orlikowski',
      author_email='mazsvr@gmail.com',
      license='BSD_3Clause',
      packages=['arcticcaptions'],
      install_requires=[
          'keras',
          'h5py',
          'Pillow'
      ],
      zip_safe=False)
