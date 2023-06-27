from setuptools import setup

setup(
    name="optimazing",
    version="0.1.0",
    description="Wrapper around scipy.optimize.minimize",
    author="Tobias Hoinka",
    author_email="thoinka@gmail.com",
    packages=["optimazing"],
    install_requires=["scipy", "numpy", "pandas"],
)
