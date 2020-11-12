import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepinpy",
    version="0.1.0",
    author="Jon Tamir",
    author_email="jtamir@utexas.edu",
    description="Deep inverse problems in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jtamir/deepinpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
