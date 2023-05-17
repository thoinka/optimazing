from setuptools import setup

setup(
    name="foptima",
    version="0.0.1",
    description="Wrapper around scipy.optimize.minimize",
    author="Tobias Hoinka",
    author_email="thoinka@gmail.com",
    packages=["foptima"],
    install_requires=["scipy", "numpy"],
)