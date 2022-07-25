# import the contents of the Rust library into the Python extension
# optional: include the documentation from the Rust module
from .pillars import emd_bulk, emd_classify, emd_classify_bulk
from .pillars import __all__, __doc__

__all__ = ["emd_bulk", "emd_classify", "emd_classify_bulk"]