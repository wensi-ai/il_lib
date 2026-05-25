import pathlib
from setuptools import setup, find_packages


PKG_NAME = "il_lib"
VERSION = "0.1"
EXTRAS = {}


def _read_file(fname):
    with pathlib.Path(fname).open() as fp:
        return fp.read()


def _fill_extras(extras):
    if extras:
        extras["all"] = list(set([item for group in extras.values() for item in group]))
    return extras


def _read_requirements(fname="requirements.txt"):
    requirements_path = pathlib.Path(fname)
    if not requirements_path.exists():
        return []
    requirements = []
    for line in requirements_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


setup(
    name=PKG_NAME,
    version=VERSION,
    author=f"{PKG_NAME} Developers",
    description="Imitation learning model library for robot manipulation",
    long_description=_read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords=["Deep Learning", "Machine Learning"],
    license="Apache License, Version 2.0",
    packages=find_packages(include=[PKG_NAME, f"{PKG_NAME}.*", "hydra_plugins"]),
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [],
        'hydra.plugins.search_path': [
            'search_path_plugin = hydra_plugins.search_path_plugin:SearchPathPlugin'
        ]
    },
    install_requires=_read_requirements(),
    extras_require=_fill_extras(EXTRAS),
    #python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
    ],
)
