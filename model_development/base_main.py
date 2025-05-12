import argparse
import logging
import sys
from .training_manager import TrainingManager

from rich.logging import RichHandler



def main() -> None:

    # Configure Rich-powered logging
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    log = logging.getLogger("main")

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--one_svm', action='store_true')
    parser.add_argument('-i', '--isolation_forest', action='store_true')
    parser.add_argument('-l', '--local_outlier', action='store_true')
    parser.add_argument('-a', '--train_augmented', action='store_true')

    args = parser.parse_args()

    if not args.one_svm and not args.isolation_forest and not args.local_outlier:
        parser.error("You must specify at least one of -o, -i or -l")

    try:
        manager = TrainingManager(args)
        manager.run()
    except Exception as e:
        log.exception("Unexpected failure")
        sys.exit(99)

if __name__ == "__main__":
    main()