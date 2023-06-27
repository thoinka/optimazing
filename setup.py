from setuptools import setup

setup(
    name="optimazing",
    version="0.1.0",
    description="Wrapper around scipy.optimize.minimize",
    author="Tobias Hoinka",
    author_email="thoinka@gmail.com",
    url="https://github.com/thoinka/optimazing",
    keywords=["Optimization", "Scipy", "Minimization"],
    packages=["optimazing"],
    install_requires=["scipy", "numpy", "pandas"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
