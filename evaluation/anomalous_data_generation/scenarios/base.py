from abc import ABC, abstractmethod
from typing import List, Tuple
import logging
from ..utils import write_to_csv

SEED_NUMBER = 42

class Scenario(ABC):
    """
    Abstract base class for scenario generators.
    """
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.seed = SEED_NUMBER

    @abstractmethod
    def generate(self) -> List[Tuple[float, float]]:
        """
        Generate scenario data as a list of (start, end) pairs.
        """
        pass

    def write(self, filename: str) -> None:
        """
        Generate scenario data and write to a CSV file.
        """
        
        data = self.generate()
        write_to_csv(data, filename)
        self.logger.info("Scenario data written to %s", filename)
