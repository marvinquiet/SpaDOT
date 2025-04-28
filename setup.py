from setuptools import setup, find_packages

setup(
    name="SpaDOT",
    version="0.1.0",
    description="Package for paper: Optimal transport modeling uncovers spatial domain dynamics in spatiotemporal transcriptomics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Wenjing Ma",
    author_email="mawenjing1993@gmail.com",
    url="https://http://marvinquiet.github.io/SpaDOT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add your dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)