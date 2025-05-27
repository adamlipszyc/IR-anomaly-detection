import argparse
import logging
import sys
from .training_manager import TrainingManager

from rich.logging import RichHandler

def parse_list_and_int(values):
    allowed_str_values = {"none", "shift", "noise", "magnitude"}
    if len(values) < 2:
        raise argparse.ArgumentTypeError(
            "You must provide at least one string followed by an integer."
        )

    *str_values, last = values

    try:
        int_value = int(last)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"The last value must be an integer, but got: '{last}'"
        )
    
    for s in str_values:
        if not isinstance(s, str) or s.isdigit():
            raise argparse.ArgumentTypeError(
                f"All values except the last must be strings (got: {str_values})"
            )
        if s.lower() not in allowed_str_values:
            raise argparse.ArgumentTypeError(
                f"Invalid string value: '{s}'. Allowed values are: {sorted(allowed_str_values)}"
            )

    return str_values, int_value

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
    parser.add_argument('-a', '--autoencoder', action='store_true')
    parser.add_argument('-h', '--hyperparameter_tuning', action='store_true')
    parser.add_argument('--lr', type=float, help="Learning rate for the model")
    parser.add_argument('--batch_size', type=int, help="Batch size for training the model")
    parser.add_argument('--num_epochs', type=int, help="Number of epochs for training the model")
    parser.add_argument(
        '--train_augmented',
        nargs='+',
        type=str,
        help='A list of strings followed by an integer',
    )

    parser.add_argument('--encoder', type=str, choices=["autoencoder", "pca"], help="Which encoder to use for hybrid models")
    parser.add_argument('--encoding_dim', type=int, help="Number of dimensions to encode the data to")
    # parser.add_argument('-s', '--split', type=int, choices=range(1,6), required=True, help="Which data split to use (1-5)")
    parser.add_argument('-e', '--train_ensemble', action='store_true')


    args = parser.parse_args()

    args.model_args = {}

    if args.lr:
        args.model_args["lr"] = args.lr
    if args.encoding_dim:
        args.model_args["encoding_dim"] = args.encoding_dim
    if args.batch_size:
        args.model_args["batch_size"] = args.batch_size
    if args.num_epochs:
        args.model_args["num_epochs"] = args.num_epochs

    # Ensure both --encoder and --encoding_dim are specified together
    if (args.encoder is not None) and (args.encoding_dim is None):
        parser.error("Both --encoder and --encoding_dim must be specified together.")

    # Convert the last element to int, rest stay as str
    if args.train_augmented is not None:
        args.augment_techniques, args.augment_factor = parse_list_and_int(args.train_augmented)
    else:
        args.augment_techniques, args.augment_factor = None, None

    if not args.one_svm and not args.isolation_forest and not args.local_outlier and not args.autoencoder:
        parser.error("You must specify at least one of -o, -i, -l or -a(int)")

    try:
        manager = TrainingManager(args)
        manager.run()
    except Exception as e:
        log.exception("Unexpected failure")
        sys.exit(99)

if __name__ == "__main__":
    main()