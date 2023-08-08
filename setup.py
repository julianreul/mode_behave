from setuptools import setup

setup(name='mode_behave',
      version='1.0.13',
      description='Estimation and simulation of discrete choice models',
      author='Julian Reul',
      author_email='j.reul@fz-juelich.de',
      url='https://github.com/julianreul/mode_behave',
      license='MIT',
      packages=['mode_behave_public'],
      install_requires=[
          'Pillow==9.5.0',
          'matplotlib==3.7.1',
          'numba==0.56.4',
          'numpy==1.23.5',
          'pandas==2.0.1',
          'pickleshare==0.7.5',
          'scikit-learn==1.2.2',
          'scikit-optimize==0.9.0',
          'scipy==1.10.1',
          'seaborn==0.12.2'
          ],    
      include_package_data=True,
      zip_safe=False)