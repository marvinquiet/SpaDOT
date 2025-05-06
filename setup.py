from setuptools import setup, find_packages

setup(
    name="SpaDOT",
    version="1.0.0-beta",
    description="Package for paper: Optimal transport modeling uncovers spatial domain dynamics in spatiotemporal transcriptomics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Wenjing Ma",
    author_email="mawenjing1993@gmail.com",
    url="https://http://marvinquiet.github.io/SpaDOT",
    packages=find_packages(),
    package_data={
        "SpaDOT.utils.OT_loss": ["libot.so"],  # Include the .so file
        "SpaDOT": ["config.yaml"] # Include the config.yaml file
    },
    install_requires=[
        "torch==2.5.0",
        "torchvision",
        "torchaudio",
        # "torch_geometric==2.6.1", # install after pyg-lib, torch-sparse and torch-scatter
        "anndata==0.9.1",
        "scanpy==1.9.8",
        "numpy<2.0.0", # compatible with scanpy 1.9.8
        "wot", # Optimal transport library
        "pandas",
        "scipy", 
        "scikit-learn",  # sklearn is part of scikit-learn
        "matplotlib",
        "seaborn",
        "tqdm",
        "pyyaml",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "SpaDOT=SpaDOT.cli:main",  # Expose the CLI via the `SpaDOT` command
        ],
    },
)
