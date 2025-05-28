from .system import SystemModel, State
from .timeseries import Cycle
from .utils import defaults, reference_frames, color_palette, utils
import logging

# Configure logging (usually done once in your main module or __init__.py)
logging.basicConfig(level=logging.INFO)  # or DEBUG for more detail
logger = logging.getLogger(__name__)  # Use module-specific logger