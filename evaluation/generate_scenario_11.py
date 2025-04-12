import random
import csv
from generate_scenario_10 import NoPossibleTradeException, IncorrectMatchingException


def generate_real_positions(num_positions=20, value_range=(-500, 500)):
    return [
        (v := random.randint(*value_range) // 10 * 10, v)  # round to nearest 10
        for _ in range(num_positions)
    ]

def random_split_with_candidates(total, candidates, start_values, spread=False):
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
    min_spread = random.randint(2, 5) #Randomise the lower bound
    max_spread = random.randint(5, 10) # Randomise the upper bound -
    current_spread = random.randint(min_spread, max_spread)
    for index, candidate in enumerate(candidates):
        if spread and prevIndex and index - prevIndex < current_spread:
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
    if available_trade > 0:
        for candidate in candidates:
            possible_trade_amount_left = candidate_max_trades[candidate] - splits[candidate]
            max_trade = min(available_trade, possible_trade_amount_left)
            splits[candidate] += max_trade
            available_trade -= max_trade
            if available_trade == 0:
                break
    

    return splits




def generate_scenario():
    total_positions = random.randint(40, 80)
    gap_size = random.randint(3, 10)

    positions = generate_real_positions(total_positions)

    start_values = [start for start, _ in positions]

    flip = random.choice([True, False])  # If True: positive at top, negative at bottom

    top_section_index = total_positions // 4 - gap_size
    top_range = range(0, top_section_index)
    bottom_section_index = total_positions // 4 + gap_size
    bottom_range = range(bottom_section_index, total_positions)


    # Extract trading candidates
    top_candidates = [i for i in top_range if (start_values[i] < 0 if not flip else start_values[i] > 0)]
    bottom_candidates = [i for i in bottom_range if (start_values[i] > 0 if not flip else start_values[i] < 0)]

    top_total_capacity = sum(abs(start_values[i]) for i in top_candidates)
    bottom_total_capacity = sum(abs(start_values[i]) for i in bottom_candidates)

    max_trade = min(top_total_capacity, bottom_total_capacity)
    if max_trade == 0:
        raise NoPossibleTradeException()  # No trading possible

    total_trade_amount = random.randint(1, max_trade)

    # Randomly assign trade amounts to candidates
    top_trades = random_split_with_candidates(total_trade_amount, top_candidates, start_values)
    bottom_trades = random_split_with_candidates(total_trade_amount, bottom_candidates, start_values, spread=True)

    # Ensure trade sums match (by trimming larger side if needed)
    top_trade_sum = sum(top_trades.values())
    bottom_trade_sum = sum(bottom_trades.values())

    if top_trade_sum != bottom_trade_sum:
        raise IncorrectMatchingException()

    # Apply the trades
    for i, amt in top_trades.items():
        start = positions[i][0]
        end = start + amt if (not flip and start < 0) else start - amt
        positions[i] = (start, end)

    for i, amt in bottom_trades.items():
        start = positions[i][0]
        end = start - amt if (not flip and start > 0) else start + amt
        positions[i] = (start, end)

    positions.append((-9999999999999, -99999999999999))
    return positions


def save_scenario_to_csv(scenario, filename="evaluation/data/generated_scen_11.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Start", "End"])
        for start, end in scenario:
            writer.writerow([start, end])


# Example usage
if __name__ == "__main__":
    # random.seed(42)
    scenario = generate_scenario()
    save_scenario_to_csv(scenario)
    print("CSV written to evaluation/data/generated_scen_11.csv")
