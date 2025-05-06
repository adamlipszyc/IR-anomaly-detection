import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from .generators import DataGenerator
from .scenarios.scen_4 import Scenario4
from .scenarios.scen_9 import Scenario9
from .scenarios.scen_10 import Scenario10
from .scenarios.scen_11 import Scenario11
from .scenarios.scen_perm import ScenarioPerm
from .scenario_exceptions import NoPossibleTradeException, IncorrectMatchingException

def make_summary(title: str, counts: dict) -> None:
    console = Console()
    table = Table(title=title)
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    for label, count in counts.items():
        table.add_row(label, str(count))
    console.print(table)


def main() -> None:
    # Logging config with Rich
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    log = logging.getLogger("main")

    parser = argparse.ArgumentParser(description="Generate training data using predefined scenarios.")
    parser.add_argument("-s", "--simple", type=int, help="Generate N simple samples")
    parser.add_argument("-c", "--complex", type=int, help="Generate N complex batches")
    parser.add_argument("--output", type=str, default="generated.npy", help="Output .npy file path")
    parser.add_argument("--with-real", action="store_true", help="Include real positions in samples")

    args = parser.parse_args()

    if not args.simple and not args.complex:
        parser.error("Must provide one of -s (simple) or -c (complex) with number of samples")

    # Prepare scenario instances
    scenarios = [Scenario4(), Scenario9(), Scenario10(), Scenario11(), ScenarioPerm()]  # Add Scenario9, Scenario10, etc. if needed


    generator = DataGenerator(
        scenarios=scenarios,
        num_simple=args.simple if args.simple else 1000,
        complex_samples=args.complex if args.complex else 100,
        logger=log
    )

    stats: dict = {}

    try:
        if args.simple:
            log.info(f"Generating {args.simple} simple samples...")
            generator.generate_simple()
            make_summary("Simple Generation Summary", {"Samples": args.simple})

        if args.complex:
            log.info(f"Generating {args.complex} complex batches...")
            generator.generate_complex()
            make_summary("Complex Generation Summary", {"Complex Samples": args.complex * 10})

    except (NoPossibleTradeException, IncorrectMatchingException) as e:
        log.error(f"Scenario generation failed: {e}")
        sys.exit(1)
    except Exception as e:
        log.exception("Unexpected failure")
        sys.exit(99)


if __name__ == "__main__":
    main()
