from setuptools import setup

setup(
    name="simple_driving",
    version='0.0.1',
    install_requires=['gymnasium',
                      'pybullet',
                      'numpy',
                      'matplotlib',
                      'torch'],
    package_data={'simple_driving': ['resources/*.urdf']}
)
