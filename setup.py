from setuptools import setup

setup(name='mode_behave',
      version='1.0.5',
      description='Estimation and simulation of discrete choice models',
      author='Julian Reul',
      author_email='j.reul@fz-juelich.de',
      url='https://github.com/julianreul/mode_behave',
      license='MIT',
      packages=['mode_behave_public'],
      include_package_data=True,
      zip_safe=False)