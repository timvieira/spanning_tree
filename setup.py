from setuptools import setup

setup(name='spanning_tree',
      author='Tim Vieira',
      description='A reference implementation of the matrix-tree theorem with applications to nonprojective spanning tree models in natural language processing.',
      version='1.0',
      install_requires=[
          'arsenal',
      ],
      dependency_links=[
          'https://github.com/timvieira/arsenal.git',
      ],
      packages=['spanning_tree'],
)
