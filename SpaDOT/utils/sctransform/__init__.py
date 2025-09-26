"""
@marvinquiet:

This package is from Stereopy, originally designed for using `Seurat` flavor SCTransform.
I have made certain modifications for my personal usage.


Coding this python-version SCTransform after learning from these R-packages:
    https://github.com/satijalab/seurat
    https://github.com/satijalab/sctransform

The R-package SCTransform is described in:
    https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1874-1
"""
from .sctransform import SCTransform
__all__ = ['SCTransform']
