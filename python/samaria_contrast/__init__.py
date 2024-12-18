from .contrast import ContrastiveLearningSystem
from .config import SamariaConfig

try:
    from ._samaria_bindings import *
except ImportError:
    import warnings

    warnings.warn("Could not import C++ bindings. Some functionality may be limited.")

__version__ = "1.0.0"
__all__ = ["ContrastiveLearningSystem", "SamariaConfig"]
