
# main.py
import sys
import argparse
import logging

from rich.logging import RichHandler


from .config import FILE_SCEN_4, FILE_SCEN_9, FILE_SCEN_10, FILE_SCEN_11, FILE_SCEN_PERM
from .scen_4 import Scenario4
from .scen_9 import Scenario9
from .scen_10 import Scenario10
from .scen_11 import Scenario11
from .scen_perm import ScenarioPerm
from ..scenario_exceptions import NoPossibleTradeException, IncorrectMatchingException
from ....log.utils import make_summary

def make_scenario_summary(stats: dict) -> None:
    make_summary("Scenario Generation Summary", stats)

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
            Scenario10().write(str(FILE_SCEN_10))
            stats["Scenario 10"] = "Success"
        elif args.scenario == 11:
            log.info("Generating Scenario 11...")
            Scenario11().write(str(FILE_SCEN_11))
            stats["Scenario 11"] = "Success"
        elif args.permutations:
            log.info("Generating Scenario permutations...")
            ScenarioPerm().write(str(FILE_SCEN_PERM))
            stats["Scenario Permutations"] = "Success"
        else:
            parser.error("Please specify a scenario (--scenario) or --permutations")

        make_scenario_summary(stats)        
    except NoPossibleTradeException as e:
        log.error("Domain error: %s", e)
        stats["Error"] = str(e)
        make_scenario_summary(stats)
        sys.exit(1)
    except IncorrectMatchingException as e:
        log.error("Matching error: %s", e)
        stats["Error"] = str(e)
        make_scenario_summary(stats)
        sys.exit(2)
    except OSError as e:
        log.error("I/O error: %s", e)
        stats["Error"] = str(e)
        make_scenario_summary(stats)
        sys.exit(3)
    except Exception as e:
        log.exception("Unexpected error")
        stats["Error"] = "Unexpected"
        make_scenario_summary(stats)
        sys.exit(99)


if __name__ == "__main__":
    main()
