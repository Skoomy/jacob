import logging
import sys


__version__ = "0.0.1"
__author__ = "Team Artometrix"
__email__ = "contact@artometrix.com"


format = (
    "[%(asctime)s][%(levelname)s][%(filename)s][%(module)s.%(funcName)s]" " %(message)s"
)

datefmt = "%Y-%m-%d %H:%M:%S"


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format=format,
    datefmt=datefmt,
)
logger = logging.getLogger(__name__)
