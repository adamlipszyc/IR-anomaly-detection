import random
from typing import List, Tuple, Dict
from .base import Scenario
from ..utils import catch_and_log, generate_real_positions, apply_trades
from ..scenario_exceptions import NoPossibleTradeException, IncorrectMatchingException

class SharedScenario(Scenario):

    def __init__(self):
        super().__init__()

    @catch_and_log(Exception, "Distributing top down")
    def distribute_top_down(
            self,
            available_trade: int, 
            candidates: List[int], 
            candidate_max_trades, 
            splits):
        splits = splits.copy()
        if available_trade > 0:
            for candidate in candidates:
                possible_trade_amount_left = candidate_max_trades[candidate] - splits[candidate]
                max_trade = min(available_trade, possible_trade_amount_left)
                splits[candidate] += max_trade
                available_trade -= max_trade
                if available_trade == 0:
                    return splits 
        
        return splits 

    @catch_and_log(Exception, "Randomly distributing amongst candidates")
    def random_split_with_candidates(
            self,
            total: int, 
            candidates: List[int], 
            start_values: List[int], 
            spread=False):
        """
        Randomly distributes 'total' units among given 'candidates' such that:
        - Every candidate may receive zero or more units
        - Spread determines whether to spread out selection among candidates 
        - The total distributed exactly equals 'total'
        - The amount assigned to each candidate is less than the absolute value of their 'start' position
        - Returns a dictionary mapping candidate indices to their assigned amount
        """
        num = len(candidates)
        if num == 0 or total == 0:
            return {}

        # Ensure that the maximum trade amount per candidate is constrained by their absolute 'start' value
        candidate_max_trades = {candidate: min(abs(start_values[candidate]), total) for candidate in candidates}

        # Randomly shuffle the candidates and try to distribute the total amount
        available_trade = total
        splits = {}

        # Ensure we don't assign more than the allowed max to each candidate
        prevIndex = None
        min_spread = random.randint(3, 5) #Randomise the lower bound
        max_spread = random.randint(5, 7) # Randomise the upper bound -
        current_spread = random.randint(min_spread, max_spread)
        for index, candidate in enumerate(candidates):
            if spread and (prevIndex is not None) and index - prevIndex < current_spread:
                continue
            # Max trade is the minimum of the remaining available trade and the max amount the candidate can trade
            max_trade_for_candidate = candidate_max_trades[candidate]

            trade_amount = random.randint(0, max_trade_for_candidate)

            # Ensure that we don't exceed the remaining available trade
            trade_amount = min(trade_amount, available_trade)

            # Assign the trade amount
            splits[candidate] = trade_amount
            available_trade -= trade_amount
            if spread and trade_amount > 0:
                prevIndex = index
                current_spread = random.randint(min_spread, max_spread)

            if available_trade == 0:
                break
        

        # If there is any remaining trade to distribute with spread, assign it by maximising the
        # trade value of already selected candidates
        if spread and available_trade > 0:
            traded_candidates = [index for index in splits.keys() if splits[index] > 0]
            for candidate in traded_candidates:
                possible_trade_amount_left = candidate_max_trades[candidate] - splits[candidate]
                max_trade = min(available_trade, possible_trade_amount_left)
                splits[candidate] += max_trade
                available_trade -= max_trade
                if available_trade == 0:
                    break

                    
    
        # If there is any remaining trade to distribute, assign it top down
        splits = self.distribute_top_down(available_trade, candidates, candidate_max_trades, splits)    

        return splits


    @catch_and_log(Exception, "Extracting trading candidates")
    def extract_trading_candidates(
            self,
            top_section_index: int,
            bottom_section_index: int,
            total_positions: int,
            flip: bool,
            start_values: List[int]):
        """
        Extract trading candidates depending on sign
        """
        top_candidates: List[int] = [
            i for i in range(0, top_section_index)
            if (start_values[i] < 0 if not flip else start_values[i] > 0)
        ]
        bottom_candidates: List[int] = [
            i for i in range(bottom_section_index, total_positions)
            if (start_values[i] > 0 if not flip else start_values[i] < 0)
        ]
        return top_candidates, bottom_candidates
    
    def generate_scenario(self, total_positions_range, gap_size_range, scenario10: bool):
        total_positions = random.randint(*total_positions_range)
        gap_size = random.randint(*gap_size_range)

        positions: List[Tuple[int, int]] = generate_real_positions(total_positions)
        start_values: List[int] = [start for start, _ in positions]

        flip = random.choice([True, False])  # If True: positive at top, negative at bottom

        divisor = 4
        if scenario10:
            divisor = 2

        half = total_positions // divisor
        top_section_index = half - gap_size
        bottom_section_index = half + gap_size

        top_candidates, bottom_candidates = self.extract_trading_candidates(
            top_section_index, 
            bottom_section_index, 
            total_positions, 
            flip, 
            start_values
            )

        top_total_capacity = sum(abs(start_values[i]) for i in top_candidates)
        bottom_total_capacity = sum(abs(start_values[i]) for i in bottom_candidates)
        max_trade = min(top_total_capacity, bottom_total_capacity)
        if max_trade == 0:
            raise NoPossibleTradeException()  # No trading possible

        total_trade_amount = random.randint(1, max_trade)

        # Randomly assign trade amounts to candidates
        top_trades = self.random_split_with_candidates(
            total_trade_amount,
            top_candidates,
            start_values,
        )

        
        bottom_trades = self.random_split_with_candidates(
            total_trade_amount,
            bottom_candidates,
            start_values,
            spread=(not scenario10)
        )

        # Ensure trade sums match
        top_trade_sum = sum(top_trades.values())
        bottom_trade_sum = sum(bottom_trades.values())
        if top_trade_sum != bottom_trade_sum:
            raise IncorrectMatchingException()

        # Apply the trades
        positions = apply_trades(top_trades, positions)
        positions = apply_trades(bottom_trades, positions)

        # Cast to float tuples to conform with base signature
        return [(float(start), float(end)) for start, end in positions]
