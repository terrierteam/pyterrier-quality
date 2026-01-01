import sys
from setuptools import setup, find_packages

def get_version(rel_path):
    import os
    for line in open(rel_path):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


requirements = []
with open('requirements.txt', 'rt') as f:
    for req in f.read().splitlines():
        if req.startswith('git+'):
            pkg_name = req.split('/')[-1].replace('.git', '')
            if "#egg=" in pkg_name:
                pkg_name = pkg_name.split("#egg=")[1]
            requirements.append("%s @ %s" % (pkg_name, req))
        else:
            requirements.append(req)

setup(
    name="pyterrier-quality",
    version=get_version("pyterrier_quality/__init__.py"),
    author="Sean MacAvaney",
    author_email='sean.macavaney@glasgow.ac.uk',
    description="Content Quality Estimation for PyTerrier",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/terrierteam/pyterrier-quality",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "onnx": ["onnx", "onnxruntime", "platformdirs"]
    },
    python_requires='>=3.10',
    entry_points={
        'pyterrier.artifact': [
            'quality_score_cache.numpy = pyterrier_quality:QualCache',
        ],
    },
)
