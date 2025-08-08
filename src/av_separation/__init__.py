try:
    from .separator import AVSeparator
    _separator_available = True
except ImportError:
    _separator_available = False

try:
    from .models import AVSeparationTransformer
    _models_available = True
except ImportError:
    _models_available = False

from .config import SeparatorConfig

try:
    from .version import __version__
except ImportError:
    __version__ = "1.0.0"

__all__ = ["SeparatorConfig", "__version__"]

if _separator_available:
    __all__.append("AVSeparator")
    
if _models_available:
    __all__.append("AVSeparationTransformer")