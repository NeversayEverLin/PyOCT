import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyOCT", # Replace with your own username
    version="2.0.7",
    author="Yuechuan Lin",
    author_email="linyuechuan1989@gmail.com",
    description="OCT imaging reconstruction on spectral-domain optical coherence tomography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeversayEverLin/PyOCT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

