import random
import logging
from typing import List, Tuple
from .shared import SharedScenario
from log.utils import catch_and_log

class Scenario10(SharedScenario):
    """
    Scenario 10 generator: random matching of real positions with equal trades top and bottom.
    """
    def __init__(
        self,
        total_positions_range: Tuple[int, int] = (60, 80),
        gap_size_range: Tuple[int, int] = (20, 40),
    ) -> None:
        super().__init__()
        self.total_positions_range: Tuple[int, int] = total_positions_range
        self.gap_size_range: Tuple[int, int] = gap_size_range
        self.logger = logging.getLogger(self.__class__.__name__)

    

    @catch_and_log(Exception, "Generating test case")
    def generate(self) -> List[Tuple[float, float]]:
        """
        Generate scenario data as a list of (start, end) pairs.
        """
        # random.seed(self.seed)
        return self.generate_scenario(
            self.total_positions_range, 
            self.gap_size_range, 
            scenario10=True)
        
# Example usage
# if __name__ == "__main__":
#     scenario = Scenario10()
#     scenario.write(filename=FILENAME)
#     print(f"Scenario 10 written to CSV: {FILENAME}")

