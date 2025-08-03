import pathlib
import pkg_resources
from setuptools import setup, find_packages


PKG_NAME = "il_lib"
VERSION = "0.1"
EXTRAS = {}


def _read_file(fname):
    with pathlib.Path(fname).open() as fp:
        return fp.read()


def _read_install_requires():
    with pathlib.Path("requirements.txt").open() as fp:
        return [
            str(requirement) for requirement in pkg_resources.parse_requirements(fp)
        ]


def _fill_extras(extras):
    if extras:
        extras["all"] = list(set([item for group in extras.values() for item in group]))
    return extras


setup(
    name=PKG_NAME,
    version=VERSION,
    author=f"{PKG_NAME} Developers",
    description="Imitation learning model library for robot manipulation",
    long_description=_read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords=["Deep Learning", "Machine Learning"],
    license="Apache License, Version 2.0",
    packages=find_packages(include=f"{PKG_NAME}.*"),
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [],
        'hydra.plugins.search_path': [
            'search_path_plugin = il_lib.hydra_plugins.search_path_plugin:SearchPathPlugin'
        ]
    },
    install_requires=_read_install_requires(),
    extras_require=_fill_extras(EXTRAS),
    #python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
    ],
)