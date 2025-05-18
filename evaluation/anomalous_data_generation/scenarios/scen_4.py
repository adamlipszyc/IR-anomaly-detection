import random
import logging
from typing import List, Tuple
from .base import Scenario
from ..utils import generate_real_positions, generate_random_zero_pairs, round_to_nearest_hundred, flip_data
from ....log.utils import catch_and_log

COUNTER_OFFSET = 300
LOWER_BOUND = 50

class Scenario4(Scenario):
    """
    Scenario 4 generator: fake trades absorbed by real positions with counter-trade.
    """
    def __init__(self, num_trades: int = 10, counter_offset: int = 300):
        super().__init__()
        self.num_trades = num_trades
        self.counter_offset = counter_offset
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @catch_and_log(Exception, "generating fake block")
    def generate_fake_trade_block(
        self,
        num_trades: int = 10
    ) -> Tuple[List[Tuple[int, int]], int]:
        """Generate fake traded positions rounded to the nearest 100 with end = 0."""
        trade_value_choices = [100 * i for i in range(1, 5)]  # 100 to 400
        block = []
        total = 0
        for _ in range(num_trades):
            val = random.choice(trade_value_choices)
            block.append((val, 0))
            total += val
        return block, total
    
    @catch_and_log(Exception, "generating counter trade")
    def generate_counter_trade(
        self,
        total_value: int
    ) -> List[Tuple[int, int]]:
        """Return a counter-trade that offsets the traded positions."""
        counter_trade_val = random.randint(total_value - COUNTER_OFFSET, total_value + COUNTER_OFFSET)
        val = round_to_nearest_hundred(counter_trade_val)
        return [(-val, -val)]

    @catch_and_log(Exception, "adjusting fake positions")
    def adjust_fake_trades(
        self,
        fake_trades: List[Tuple[int, int]],
        total_fake_sum: int,
        negative_real_positions_sum: int
    ) -> List[Tuple[int, int]]:
        """Ensure that the sum of the fake trades can be absorbed by the real positions."""
        if total_fake_sum > negative_real_positions_sum:
            # Calculate the excess amount
            excess = total_fake_sum - negative_real_positions_sum

            total_adjusted = 0 
            i = len(fake_trades) - 1
            current_adjustment = fake_trades[i][0]
            while total_adjusted + current_adjustment < excess:
                total_adjusted += current_adjustment
                fake_trades[i] = (fake_trades[i][0], current_adjustment)
                i -= 1
                current_adjustment = fake_trades[i][0]
            
            leftover = excess - total_adjusted
            fake_trades[i] = (fake_trades[i][0], leftover)


        return fake_trades

    @catch_and_log(Exception, "adjusting real positions")
    def adjust_real_positions_with_fake_trades(self,
        real_positions: List[Tuple[int, int]],
        fake_trades: List[Tuple[int, int]],
        fake_trade_sum: int
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Adjust some of the real negative positions to absorb the fake trades.
        Ensures that no 'End' becomes greater than 0 and no position becomes more negative.
        The adjustment is done using random values that sum up to the fake trade sum.
        """
        # Filter for negative positions
        negative_real_positions = [p for p in real_positions if p[0] < 0]
        
        # Randomly choose a subset of negative positions to absorb part of the fake trades
        num_to_adjust = random.randint(1, len(negative_real_positions))  # Adjust 1 to all negative positions
        total_to_absorb = fake_trade_sum
        
        # Randomly select num_to_adjust unique negative positions
        selected_positions = random.sample(negative_real_positions, num_to_adjust)
        
        total_adjusted = 0
        
        # Apply adjustments to the selected positions
        for i in range(num_to_adjust - 1):
            position = selected_positions[i]

            # The remaining amount that needs to be absorbed
            remaining_to_absorb = total_to_absorb - total_adjusted
            
            # If the remaining adjustment is smaller than or equal to the position's magnitude, adjust it
            max_adjustment = min(abs(position[1]), remaining_to_absorb)
            
            if max_adjustment > 0:
                lowerbound = LOWER_BOUND if max_adjustment > LOWER_BOUND else 1
                adjustment = random.randint(lowerbound, max_adjustment)
                total_adjusted += adjustment
                real_positions[real_positions.index(position)] = (position[0], position[1] + adjustment)
            else:
                # Once we have absorbed enough, stop adjusting further and break
                break
        
        # Apply the last adjustment to make sure the sum of adjustments matches total_to_absorb
        remaining_to_absorb = total_to_absorb - total_adjusted
        if remaining_to_absorb > 0:
            
            # Add the remaining amount to the last selected position
            last_position = selected_positions[-1]
            if remaining_to_absorb > abs(last_position[1]):
                fake_trades = self.adjust_fake_trades(fake_trades, fake_trade_sum, total_adjusted + abs(last_position[1]))
            remaining_to_absorb = min(remaining_to_absorb, abs(last_position[1]))
            real_positions[real_positions.index(last_position)] = (
                last_position[0], last_position[1] + remaining_to_absorb
            )
        
        return fake_trades, real_positions

    @catch_and_log(Exception, "Generating test case")
    def generate(self) -> List[Tuple[float, float]]:
        # random.seed(self.seed)
        data = []
        
        num_trades = random.randint(3, 10)

        # Generate traded block and store total amount traded
        fake_trades, total_trade_value = self.generate_fake_trade_block(num_trades)

        # Generate a real position block
        real_positions = generate_real_positions(10, 30)
        
        # Adjust some of the real positions to absorb fake trades, ensuring 'End' is never > 0
        fake_trades, real_positions = self.adjust_real_positions_with_fake_trades(real_positions, fake_trades, total_trade_value)
        
        # Add fake trades
        data.extend(fake_trades)

        # Insert a few zeros between traded block and the counter-trade
        data.extend(generate_random_zero_pairs(3, 10))

        # Add counter-trade
        data.extend(self.generate_counter_trade(total_trade_value))

        # Add tail of zeroes and real positions
        tail = generate_random_zero_pairs(3, 10) + real_positions + generate_random_zero_pairs(3, 10)

        data.extend(tail)

        data = flip_data(data)
       
        return data

# if __name__ == "__main__":
#     data = generate_test_case(num_blocks=10)
#     write_to_csv(data, filename=FILENAME)
