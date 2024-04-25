from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("translocator-models")
except PackageNotFoundError:
    # package is not installed
    pass
