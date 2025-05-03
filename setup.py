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
        "torch==2.0.1",
        "torch_geometric==2.6.1",
        "anndata==0.9.1",
        "scanpy==1.9.8",
        "numpy==1.22.4",
        "pandas==1.3.5",
        "scikit-learn==1.3.0",  # sklearn is part of scikit-learn
        "scipy==1.10.1",
        "matplotlib==3.6.3",
        "seaborn==0.11.2",
        "wot==1.0.8"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
