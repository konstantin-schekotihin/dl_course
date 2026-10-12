"""
Helper code for the AAU ML/DL course notebooks.

Notebooks import one course module as `hp`, e.g. `from dl_course import deep_learning as hp`.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dl-course")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "unknown"
