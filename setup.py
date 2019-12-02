import setuptools
import versioneer

with open("README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="dmarket",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Reinforced BEYBs",
    author_email="zhy@tuta.io",
    description="Fast double auction market engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhy0/dmarket_rl",
    packages=['dmarket'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license_file="LICENSE",
    python_requires='>=3.6',
    install_requires=requirements
)
