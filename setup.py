from setuptools import setup

setup(
   name='JapaneseRecognizer',
   version='0.1',
   description='A useful module',
   author='Adrien Logut',
   author_email='newsadrien@gmail.com',
   packages=['JR'],  #same as name
   install_requires=['torch', 'wget', 'numpy', 'matplotlib', 'pillow', 'scipy', 'scikit-image'], #external packages as dependencies
)
