"""
fastcpd: Fast change point detection using sequential gradient descent

This package provides efficient algorithms to identify points in time where the
statistical properties of a sequence of observations change.
"""

__version__ = "0.18.1"

from fastcpd.fastcpd import fastcpd
from fastcpd import segmentation
from fastcpd import metrics
from fastcpd import datasets
from fastcpd import visualization

__all__ = ["fastcpd", "segmentation", "metrics", "datasets", "visualization", "__version__"]
