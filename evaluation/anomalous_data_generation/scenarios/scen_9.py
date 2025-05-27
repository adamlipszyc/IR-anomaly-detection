import random
import logging
from typing import List, Tuple
from .base import Scenario
from ..utils import generate_random_zero_pairs, round_to_nearest_hundred, flip_data, generate_flag_array
from log.utils import catch_and_log


class Scenario9(Scenario):
    """
    Scenario 9 generator: two mismatched blocks balanced with zeros and flags.
    """
    def __init__(self, num_trades: int = 5):
        super().__init__()
        self.num_trades = num_trades
        self.logger = logging.getLogger(self.__class__.__name__)

    @catch_and_log(Exception, "Generating trade block")
    def generate_trade_block(self, num_trades: int = 5) -> Tuple[List[Tuple[int, int]], int]:
        """Generate a block of positive trade positions with End randomly less than or equal to Start."""
        trade_value_choices = [100 * i for i in range(1, 10)]  # 100 to 900
        block: List[Tuple[int, int]] = []
        total = 0
        large_negative_index = random.randint(0, num_trades - 1)
        flags = generate_flag_array(num_trades)
        for i in range(num_trades):
            start = random.choice(trade_value_choices)
            shouldTrade = flags[i]
            if shouldTrade:
                end = random.randint(0, start)
            else:
                end = start
            block.append((start, end))
            total += (start - end)
            if large_negative_index == i:
                large_value = round_to_nearest_hundred(random.randint(1000, 3000))
                block.append((-large_value, -large_value))

        return block, total

    @catch_and_log(Exception, "Balancing traded amounts")
    def balance_traded_amounts(
        self,
        positive_block: List[Tuple[int, int]],
        negative_block: List[Tuple[int, int]],
        positive_sum: int,
        negative_sum: int
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Ensure the sum of positive traded equals negative traded by reducing the excess in reverse order."""
        larger_sum = 0
        smaller_sum = 0
        larger_block = None

        # Determine which is the larger block regarding amount traded
        if positive_sum > negative_sum:
            larger_sum = positive_sum
            smaller_sum = negative_sum
            larger_block = positive_block
        else:
            larger_sum = negative_sum
            smaller_sum = positive_sum
            larger_block = negative_block

        # Balance the amounts traded so they match 
        excess = larger_sum - smaller_sum
        i = len(larger_block) - 1
        while excess != 0:
            #Skip the large negative position
            if abs(larger_block[i][0]) >= 1000:
                i -= 1
                continue

            # Figures out much we can possibly reduce by 
            reducible = larger_block[i][0] - larger_block[i][1]
            reduce_by = min(excess, abs(reducible))
            if reducible < 0 and reduce_by > 0:
                reduce_by = -reduce_by
        
            # Carry out reduction 
            larger_block[i] = (larger_block[i][0], larger_block[i][1] + reduce_by)
            excess -= abs(reduce_by)
            i -= 1

        return positive_block, negative_block

    @catch_and_log(Exception, "Generating test case")
    def generate(self) -> List[Tuple[int, int]]:
        # random.seed(self.seed)
        """Generate one full test scenario with two mismatched blocks and intermediate zero rows."""
        pos_block, positive_trade_sum = self.generate_trade_block(random.randint(3, 6))
        neg_block, negative_trade_sum = self.generate_trade_block(random.randint(3, 6))

        neg_block = [(-start, -end) for start, end in neg_block]

        pos_block, neg_block = self.balance_traded_amounts(pos_block, neg_block, positive_trade_sum, negative_trade_sum)

        # Insert random number of zeros in between blocks
        zero_rows = generate_random_zero_pairs(2, 6)

        scenario = (
            generate_random_zero_pairs(1, 6)
            + pos_block
            + zero_rows
            + neg_block
            + generate_random_zero_pairs(1, 6)
        )

        scenario = flip_data(scenario)

        return scenario
    
# Example usage:
# if __name__ == "__main__":
#     Scenario9().write(10)

