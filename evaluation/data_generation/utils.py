import csv
import random
import logging
from typing import List, Tuple, Dict, Callable, Optional, Type
from functools import wraps
from rich.logging import RichHandler

# Configure root logger to use Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)

ZERO_PAIR: Tuple[int, int] = (0, 0)

def generate_random_zero_pairs(lower: int, upper: int) -> List[Tuple[int, int]]:
    """
    Generate a random number of zero pairs between lower and upper (inclusive).
    """
    count = random.randint(lower, upper)
    logger.debug("Generating %d zero pairs", count)
    return [ZERO_PAIR for _ in range(count)]

def generate_real_positions(count: int, max_count: Optional[int] = None,
                            value_range: Tuple[int, int] = (-500, 500)) -> List[Tuple[int, int]]:
    """
    Generate real positions. If max_count is provided, select a random number between count and max_count;
    otherwise generate exactly count positions. Each as (v, v), v rounded to nearest 10.
    """
    num = random.randint(count, max_count) if max_count is not None else count
    positions: List[Tuple[int, int]] = []
    for _ in range(num):
        v = (random.randint(*value_range) // 10) * 10
        positions.append((v, v))
    logger.debug("Generated %d real positions", num)
    return positions


def generate_multiple_scenarios(generator: Callable[[], List[Tuple[int, int]]], num: int = 10) -> List[Tuple[int, int]]:
    """
    Generate multiple scenarios by calling generator repeatedly and append a delimiter after each.
    """
    scenarios: List[Tuple[int, int]] = []
    for _ in range(num):
        data = generator()
        scenarios.extend(data)
        scenarios.append((-999999999, -999999999))  # delimiter
    logger.debug("Generated %d scenarios with delimiters", num)
    return scenarios

def write_to_csv(data: List[Tuple[float, float]], filename: str) -> None:
    """
    Write data to CSV file with headers Start, End. Raises OSError on failure.
    """
    try:
        logger.info("Writing %d rows to %s", len(data), filename)
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Start", "End"])
            writer.writerows(data)
    except OSError as e:
        logger.error("Failed to write %s: %s", filename, e)
        raise


def round_to_nearest_hundred(num: float) -> int:
    """
    Simply rounds the passed number to the nearest hundred 
    """    
    return round(num / 100) * 100

def generate_flag_array(n: int) -> List[int]:
    """
    Generates a list of n binary values (0 or 1),
    ensuring that at least one value is 1.

    Args:
        n (int): Number of binary values to generate.

    Returns:
        list[int]: List of 0s and 1s with at least one 1.
    """
    if n <= 0:
        raise ValueError("n must be greater than 0")

    # Start with all random 0s or 1s
    values = [random.randint(0, 1) for _ in range(n)]

    # Ensure at least one 1
    if all(v == 0 for v in values):
        # Replace one random position with a 1
        values[random.randint(0, n - 1)] = 1

    return values


def generate_flag() -> int:
    """
    Generate a random flag, 0 or 1.
    """
    return random.randint(0, 1)


def apply_trades(trades: Dict[int, int], positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Apply trade amounts to positions and return updated positions.
    """
    logger.debug("Applying trades: %s", trades)
    updated = positions.copy()
    for idx, amt in trades.items():
        start, _ = updated[idx]
        end = start + amt if start < 0 else start - amt
        updated[idx] = (start, end)
    return updated


def convert_end_to_traded(positions) -> List[Tuple[int, float]]:
    """
    Converts a list of (start, end) tuples into a list of (start, traded) tuples
    """
    result: List[Tuple[int, float]] = []
    for position in positions:
        for j in range(0, len(position), 2):
            start = position[j]
            end = position[j + 1]
            traded = float(start) - float(end)
            result.append((start, traded))

    return result



def catch_and_log(exception: Type[Exception], action: str = "") -> Callable:
    """
    Decorator to catch specified exception, log error, and re-raise.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_logger = getattr(args[0], "logger", logger) if args else logger
            try:
                return func(*args, **kwargs)
            except exception as e:
                current_logger.error("Error %s[%s]: %s", action or func.__name__, type(e).__name__, e)
                raise
            finally:
                current_logger.info(f"{func.__name__}")
        return wrapper
    return decorator


def flip_data(data):
    flip = generate_flag()
    if flip:
        data = list(map(lambda x: (-x[0], -x[1]), data))
    
    return data

def call_until_success(func, *args, **kwargs):
    """
    Calls func(*args, **kwargs) repeatedly until it completes
    without raising. Returns the successful result.
    """
    while True:
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            # you could log e here, or inspect its type,
            # or sleep() before retrying, etc.
            continue
        else:
            # no exception: break out of the loop
            return result
