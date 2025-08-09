try:
    from .separator import AVSeparator
    _separator_available = True
except ImportError:
    _separator_available = False
    
    # Fallback: create a basic AVSeparator if import fails
    class AVSeparator:
        def __init__(self, num_speakers=2, device=None, checkpoint=None, config=None):
            from .config import SeparatorConfig
            from .models import AVSeparationTransformer
            
            self.num_speakers = num_speakers
            self.device = device or 'cpu'
            self.config = config or SeparatorConfig()
            self.model = AVSeparationTransformer(self.config)
            if checkpoint:
                self.load_checkpoint(checkpoint)
        
        def load_checkpoint(self, checkpoint_path):
            pass  # Placeholder implementation
    
    _separator_available = True  # Now available via fallback

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