import numpy as np 
import random
import csv

ZERO_PAIR = (0, 0)

def generate_random_zero_pairs(lower, upper):
    """
    Generates a random number of zero pairs in between lower and upper specified
    """
    num_zeroes = random.randint(lower, upper)
    return [ZERO_PAIR for _ in range(num_zeroes)]

def generate_real_positions(lower, upper=0, value_range=(-500, 500)):
    
    """
    Generates real positions but behaviour is dependent on number of arguments 
    specified. 
    - Only lower: 
        lower is taken as the exact number of positions 
    - Both lower and upper: 
        we generate a random number within the range (lower, upper) and 
        generate this many real positions
    """
    num_positions = random.randint(lower, upper) if upper else lower
    
    return [
        (v := random.randint(*value_range) // 10 * 10, v)  # round to nearest 10
        for _ in range(num_positions)
    ]


def generate_multiple_scenarios(scenario_generator, num=10):
    all_rows = []
    for _ in range(num):
        scenario = scenario_generator()
        all_rows.extend(scenario)
        all_rows.append([-999999999, -999999999])  # delimiter
    return all_rows


def write_to_csv(data, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Start", "End"])
        writer.writerows(data)


def round_to_nearest_hundred(num):
    """
    Simply rounds the passed number to the nearest hundred 
    """    
    return round(num / 100) * 100

def generate_flag():
    return random.randint(0, 1)


def apply_trades(trades, positions):
    for i, amt in trades.items():
        start = positions[i][0]
        end = start + amt if (start < 0) else start - amt
        positions[i] = (start, end)
    
    return positions


def convert_end_to_traded(positions):
    """
    Converts a list of (start, end) tuples into a list of (start, traded) tuples
    """
    result = []
    for position in positions:
        for j in range(0, len(position), 2):
            start = position[j]
            end = position[j + 1]
            traded = float(start) - float(end)
            result.append((start, traded))

    return result
