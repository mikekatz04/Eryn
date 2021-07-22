# from future.utils import iteritems
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

# setup version file
with open("README.md", "r") as fh:
    lines = fh.readlines()

for line in lines:
    if line.startswith("Current Version"):
        version_string = line.split("Current Version: ")[1].split("\n")[0]

with open("eryn/_version.py", "w") as f:
    f.write("__version__ = '{}'".format(version_string))

setup(
    name="Eryn",
    # Random metadata. there's more you can supply
    author_email="mikekatz04@gmail.com",
    author="Michael Katz, Nikos Karnesis",
    description="An all purpose MCMC sampler",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=version_string,
    url="https://github.com/mikekatz04/Eryn",
    packages=[
        "eryn",
        "eryn.backends",
        "eryn.moves",
        "eryn.utils",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
)
