from setuptools import setup, find_packages

setup(name='breast_cancer',
      version='0.1',
      description='The neural network to recognize breast cancer',
      url='https://github.com/lafarinio/breast_cancer',
      author='Rafał Kocoń, Robert Krzaczyński',
      author_email='226467@student.pwr.edu.pl, 226491@student.pwr.edu.pl',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'scipy',
          'numpy',
          'pandas',
          'anaconda',
          'scikit-learn'
      ],
      zip_safe=False,
      include_package_data=True,
      python_requires=">=3.6",
      entry_points={
          'console_scripts': [
              'breast-cancer = src.__main__:main'
          ]
      },)
