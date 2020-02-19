from setuptools import setup

setup(
   name='JapaneseRecognizer',
   version='0.1',
   description='A useful module',
   author='Adrien Logut',
   author_email='newsadrien@gmail.com',
   packages=['JapaneseRecognizer'],  #same as name
   install_requires=['pytorch'], #external packages as dependencies
)
