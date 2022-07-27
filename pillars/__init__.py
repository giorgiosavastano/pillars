# import the contents of the Rust library into the Python extension
# optional: include the documentation from the Rust module
from .pyllars import emd_bulk, emd_classify, emd_classify_bulk
from .pyllars import __all__, __doc__

__all__ = ["emd_bulk", "emd_classify", "emd_classify_bulk"]