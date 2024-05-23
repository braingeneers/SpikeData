from distutils.core import setup

setup(
    name="SpikeData",
    version="0.1.0",
    description="Data structure and analysis for spike data",
    py_modules=["spikedata"],
    install_requires=[
        "numpy",
        "scipy(>=1.10.0)",
        "powerlaw",
    ],
)
