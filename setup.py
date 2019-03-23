from setuptools import setup

setup(name='breast_cancer',
      version='0.1',
      description='The neural network to recognize breast cancer',
      url='https://github.com/lafarinio/breast_cancer',
      author='Rafał Kocoń, Robert Krzaczyński',
      author_email='226467@student.pwr.edu.pl, XXXXXX@student.pwr.edu.pl',
      license='MIT',
      packages=['breast_cancer'],
      install_requires=[
          'scipy',
      ],
      zip_safe=False)
