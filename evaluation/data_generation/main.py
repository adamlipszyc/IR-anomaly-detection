
# main.py
import sys
import argparse
import logging

from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table

from config import FILE_SCEN_4, FILE_SCEN_9, FILE_SCEN_10, FILE_SCEN_11
from scenarios.scen_4 import Scenario4
from scenarios.scen_9 import Scenario9
# from scenarios.scenario10 import Scenario10
# from scenarios.scenario11 import Scenario11
# from permutations import generate_examples_easy, generate_examples_repeat
from scenario_exceptions import NoPossibleTradeException, IncorrectMatchingException


def make_summary(stats: dict) -> None:
    """
    Display a summary table of generation outcomes.
    """
    console = Console()
    table = Table(title="Scenario Generation Summary")
    table.add_column("Task", style="cyan")
    table.add_column("Status", style="magenta", justify="right")
    for task, status in stats.items():
        table.add_row(task, status)
    console.print(table)


def main() -> None:
    # Configure Rich-powered logging
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    log = logging.getLogger("main")

    parser = argparse.ArgumentParser(description="Generate trading scenarios.")
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[4, 9, 10, 11],
        help="Scenario number to generate"
    )
    parser.add_argument(
        "--permutations",
        action="store_true",
        help="Generate permutation examples"
    )
    parser.add_argument(
        "--with-real",
        action="store_true",
        help="Include real positions in permutations"
    )
    args = parser.parse_args()

    stats: dict = {}
    try:
        if args.scenario == 4:
            log.info("Generating Scenario 4...")
            Scenario4().write(str(FILE_SCEN_4))
            stats["Scenario 4"] = "Success"
        elif args.scenario == 9:
            log.info("Generating Scenario 9...")
            Scenario9().write(str(FILE_SCEN_9))
            stats["Scenario 9"] = "Success"
        elif args.scenario == 10:
            log.info("Generating Scenario 10...")
            # Scenario10().write(str(FILE_SCEN_10))
            stats["Scenario 10"] = "Success"
        elif args.scenario == 11:
            log.info("Generating Scenario 11...")
            # Scenario11().write(str(FILE_SCEN_11))
            stats["Scenario 11"] = "Success"
        elif args.permutations:
            key = "Permutations"
            if args.with_real:
                key += " (with real)"
            log.info("Generating %s...", key)
            # generate_examples_easy(with_real=args.with_real)
            # generate_examples_repeat(with_real=args.with_real)
            stats[key] = "Success"
        else:
            parser.error("Please specify a scenario (--scenario) or --permutations")

        make_summary(stats)
    except NoPossibleTradeException as e:
        log.error("Domain error: %s", e)
        stats["Error"] = str(e)
        make_summary(stats)
        sys.exit(1)
    except IncorrectMatchingException as e:
        log.error("Matching error: %s", e)
        stats["Error"] = str(e)
        make_summary(stats)
        sys.exit(2)
    except OSError as e:
        log.error("I/O error: %s", e)
        stats["Error"] = str(e)
        make_summary(stats)
        sys.exit(3)
    except Exception as e:
        log.exception("Unexpected error")
        stats["Error"] = "Unexpected"
        make_summary(stats)
        sys.exit(99)


if __name__ == "__main__":

    from utils import distribute_towards_zero

    lst = [100, -100, 200, -100, -300, -100, 50, -200, 90]

    # Distribute +200 to negatives
    print(distribute_towards_zero(lst, 200))

    # Distribute -150 to positives
    print(distribute_towards_zero(lst, -150))



    # main()
