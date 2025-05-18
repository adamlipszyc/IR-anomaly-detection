import logging
import sys
from .test_data_split_generator import TestDataSplitGenerator
from log.utils import make_summary
from rich.logging import RichHandler



 # === Run ===
if __name__ == "__main__":
     # Logging config with Rich
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    log = logging.getLogger("main")
    
    try:
        stats = {}
        split_generator = TestDataSplitGenerator(stats)
        split_generator.generate()
        make_summary("Test Data Split Generation", stats)
    except:
        log.error(f"Test data split generation failed")
        sys.exit(1)
