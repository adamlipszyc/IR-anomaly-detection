import random
import logging
from typing import List, Dict, Tuple
from .base import Scenario
from ..utils import (
    generate_random_zero_pairs,
    round_to_nearest_hundred,
)
from log.utils import catch_and_log

class ScenarioPerm(Scenario):
    """
    Scenario Permutations generator: random trade blocks distributed towards zero with separators.
    """
    def __init__(
        self,
        min_blocks: int = 2,
        max_blocks: int = 5,
        block_size_range: Tuple[int, int] = (2, 10),
        zeros_between_blocks: Tuple[int, int] = (6, 10),
    ) -> None:
        super().__init__()
        self.min_blocks: int = min_blocks
        self.max_blocks: int = max_blocks
        self.block_size_range: Tuple[int, int] = block_size_range
        self.zeros_between_blocks: Tuple[int, int] = zeros_between_blocks
        self.logger = logging.getLogger(self.__class__.__name__)

    @catch_and_log(Exception, "generating trade block")
    def generate_trade_block(
        self,
        count: int,
        min_val: int = -900,
        max_val: int = 900,
        round_to_hundred: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Generate a list of `count` random integer positions in [min_val, max_val],
        guaranteed to contain at least one positive and one negative value.

        Args:
            count: how many positions to generate (must be ≥ 2).
            min_val: minimum possible value (default -900).
            max_val: maximum possible value (default 900).
            round_to_hundred: if True, round every value to the nearest 100.

        Returns:
            A shuffled list of `count` integers, containing at least one >0 and one <0.
        """
        if count < 2:
            raise ValueError("count must be at least 2 to include both a positive and a negative")

        # Generate the bulk of them
        positions = [random.randint(min_val, max_val) for _ in range(count - 2)]

        # Force at least one positive and one negative
        positions.append(random.randint(1, max_val))
        positions.append(random.randint(min_val, -1))

        positions = [(v, v) for v in positions]
        # Shuffle so they’re in random order
        random.shuffle(positions)

        if round_to_hundred:
            # Round each to nearest 100
            positions = [(v:=round_to_nearest_hundred(start), v) for start, _ in positions]

        return positions

    @catch_and_log(Exception, "distributing positions towards zero")
    def distribute_towards_zero(
        self,
        positions: List[Tuple[int, int]], 
        total: int) -> List[Tuple[int, int]]:
        """
        Distributes `total` randomly across elements of `lst`:
        - If total is positive: distributes to negative values (adds, moving them toward 0 but still < 0)
        - If total is negative: distributes to positive values (subtracts, moving them toward 0 but still > 0)

        Parameters:
            lst: List of positions.
            total (int): Amount to distribute.

        Returns:
            Modified positions with distribution applied.
        """
        if total == 0:
            return positions.copy()

        updated = positions.copy()

        if total > 0:
            # Positive total → target negative numbers
            target_indices = [i for i, val in enumerate(positions) if val[0] < 0]
            max_addable = [abs(positions[i][0]) for i in target_indices]

            if total > sum(max_addable):
                raise ValueError("Total too large: would push negatives past zero")

            allocation = [0] * len(target_indices)
            remaining = total
            while remaining > 0:
                candidates = [i for i, max_val in enumerate(max_addable) if allocation[i] < max_val]
                if not candidates:
                    break
                idx = random.choice(candidates)
                allocation[idx] += 1
                remaining -= 1

            for i, alloc in zip(target_indices, allocation):
                updated[i] = (updated[i][0], updated[i][0] + alloc)

        else:
            # Negative total → target positive numbers
            total_abs = abs(total)
            target_indices = [i for i, val in enumerate(positions) if val[0] > 0]
            max_subtractable = [positions[i][0] for i in target_indices]

            if total_abs > sum(max_subtractable):
                raise ValueError("Total too large: would push positives past zero")

            allocation = [0] * len(target_indices)
            remaining = total_abs
            while remaining > 0:
                candidates = [i for i, max_val in enumerate(max_subtractable) if allocation[i] < max_val]
                if not candidates:
                    break
                idx = random.choice(candidates)
                allocation[idx] += 1
                remaining -= 1

            for i, alloc in zip(target_indices, allocation):
                updated[i] = (updated[i][0], updated[i][0] - alloc)

        return updated


    def generate(
        self
    ) -> List[Tuple[int, int]]:
        """
        Generate one evaluation test case.

        1. Pick a random number of blocks between `min_blocks` and `max_blocks`.
        2. Generate that many trade blocks
        3. Randomly choose one block to sum its positive entries, and another (distinct)
        to sum the absolute values of its negatives. Take the smaller of those two sums.
        4. Distribute the trades on each of those two chosen blocks; 
        5. For all the other blocks, leave untouched.
        6. Stitch all blocks together into one long list of positions, inserting
        a random-length of zero-pairs between them.

        Returns:
            A single list of `(start, end)` pairs representing the full scenario.
        """
        # 1. Choose how many blocks to make
        num_blocks = random.randint(self.min_blocks, self.max_blocks)

        # 2. Build each block of raw ints
        blocks: List[List[Tuple[int, int]]] = [
            self.generate_trade_block(
                count=random.randint(*self.block_size_range),
                round_to_hundred=random.choice([True, False])
            )
            for _ in range(num_blocks)
        ]

        # 3. Pick one block for positives, one for negatives
        pos_idx = random.randrange(num_blocks)
        # ensure negative‐block is different
        neg_idx = random.choice([i for i in range(num_blocks) if i != pos_idx])

        # Sum positive values in pos‐block
        pos_sum = sum(start for start, _ in blocks[pos_idx] if start > 0)
        # Sum absolute negatives in neg‐block
        neg_sum = sum(-start for start, _ in blocks[neg_idx] if start < 0)

        
        # 4. Determine how much to actually distribute
        trade_amount = random.randint(1, min(pos_sum, neg_sum))

        # Distribute that amount across each chosen block
        pos_trades = self.distribute_towards_zero(blocks[pos_idx], -trade_amount)
        neg_trades = self.distribute_towards_zero(blocks[neg_idx], trade_amount)

        # 5.–6. Stitch everything together with zero‐pairs in between
        scenario: List[Tuple[int, int]] = []

        for i, block in enumerate(blocks):
            if i == pos_idx:
                # use the (start,end) pairs returned by distribute_trades
                scenario.extend(pos_trades)
            elif i == neg_idx:
                scenario.extend(neg_trades)
            else:
                # untouched portion
                scenario.extend(block)

            # insert a random number of zero‐pairs between blocks
            # (except after the very last one, if you prefer)
            if i < num_blocks - 1:
                scenario.extend(generate_random_zero_pairs(*self.zeros_between_blocks))

        return scenario
