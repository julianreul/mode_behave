from setuptools import setup

setup(name='mode_behave',
      version='1.0.6',
      description='Estimation and simulation of discrete choice models',
      author='Julian Reul',
      author_email='j.reul@fz-juelich.de',
      url='https://github.com/julianreul/mode_behave',
      license='MIT',
      packages=['mode_behave_public'],
      install_requires=[
          'Pillow>=9.4.0',
          'matplotlib>=3.5.1',
          'numba>=0.55.1',
          'numpy>=1.21.5',
          'pandas>=1.4.2',
          'pickleshare>=0.7.5',
          'scikit-learn>=1.0.2',
          'scikit-optimize>=0.9.0',
          'scipy>=1.7.3',
          'seaborn>=0.11.2'
          ],    
      include_package_data=True,
      zip_safe=False)