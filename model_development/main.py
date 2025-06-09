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

def parse_list_ints(values):
    return list(map(lambda x: int(x), values))

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
    parser.add_argument('--anogan', action='store_true')
    parser.add_argument('--cnn_anogan', action='store_true')
    parser.add_argument('--cnn_supervised_1d', action='store_true')
    parser.add_argument('--cnn_supervised_2d', action='store_true')
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--lr', type=float, help="Learning rate for the model")
    parser.add_argument('--batch_size', type=int, help="Batch size for training the model")
    parser.add_argument('--num_epochs', type=int, help="Number of epochs for training the model")
    parser.add_argument('--kernel_size', type=int, help="Kernel size for CNN supervised models")
    parser.add_argument('--threshold', type=float, help="Threshold to use for evaluation purposes")
    parser.add_argument('--hidden_dims', nargs='+', type=str, help='A list of ints')
    parser.add_argument('--activation', type=str, help="activation function for complex models")
    parser.add_argument('--out_channels', nargs='+', type=str, help='Channel dimensions for CNN supervised')
    parser.add_argument('--fc1_size', type=int)

    parser.add_argument('--dropout', type=float)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--bidirectional', type=str)
    parser.add_argument('--num_layers', type=int)
    # === One-Class SVM ===
    parser.add_argument('--kernel', type=str, choices=["linear", "rbf", "poly", "sigmoid"],
                        help="Kernel type to be used in the One-Class SVM")
    parser.add_argument('--nu', type=float, help="An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors (One-Class SVM)")
    parser.add_argument('--gamma', type=str, help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels. Can be 'scale', 'auto', or a float value")

    # === Isolation Forest ===
    parser.add_argument('--n_estimators', type=int, help="Number of trees in the Isolation Forest")
    parser.add_argument('--max_samples', type=float, help="Fraction of samples to draw from X to train each base estimator")
    parser.add_argument('--contamination', type=float, help="Proportion of outliers in the data (Isolation Forest)")

    # === LOF (Local Outlier Factor) ===
    parser.add_argument('--n_neighbors', type=int, help="Number of neighbors to use by default for k-neighbors queries (LOF)")
    parser.add_argument('--metric', type=str, choices=["euclidean", "manhattan"],
                        help="Distance metric for LOF")
    parser.add_argument('--lof_contamination', type=float, help="Proportion of outliers in the data (LOF)")
    # parser.add_argument('--novelty', action='store_true', help="Enable novelty detection for LOF (required for test-time prediction)")

    parser.add_argument(
        '--train_augmented',
        nargs='+',
        type=str,
        help='A list of strings followed by an integer',
    )

    parser.add_argument('--hyperparameter_tuning', action='store_true')

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
    if args.kernel_size:
        args.model_args["kernel_size"] = args.kernel_size
    if args.threshold:
        args.model_args["threshold"] = args.threshold
    
    if args.activation:
        args.model_args["activation"] = args.activation
    
    if args.fc1_size:
        args.model_args["fc1_size"] = args.fc1_size
    if args.dropout is not None:
        args.model_args["dropout"] = args.dropout
    if args.hidden_size:
        args.model_args["hidden_size"] = args.hidden_size
    if args.bidirectional:
        args.model_args["bidirectional"] = args.bidirectional == "True" 
    if args.num_layers:
        args.model_args["num_layers"] = args.num_layers


    # One-Class SVM
    if args.kernel is not None:
        args.model_args["kernel"] = args.kernel
    if args.nu is not None:
        args.model_args["nu"] = args.nu
    if args.gamma is not None:
        args.model_args["gamma"] = args.gamma

    # Isolation Forest
    if args.n_estimators is not None:
        args.model_args["n_estimators"] = args.n_estimators
    if args.max_samples is not None:
        args.model_args["max_samples"] = args.max_samples
    if args.contamination is not None:
        args.model_args["contamination"] = args.contamination

    # LOF
    if args.n_neighbors is not None:
        args.model_args["n_neighbors"] = args.n_neighbors
    if args.metric is not None:
        args.model_args["metric"] = args.metric
    if args.lof_contamination is not None:  # avoid name clash
        args.model_args["contamination"] = args.lof_contamination
    # if args.novelty is not None:
    #     args.model_args["novelty"] = args.novelty

    # Ensure both --encoder and --encoding_dim are specified together
    if (args.encoder is not None) and (args.encoding_dim is None):
        parser.error("Both --encoder and --encoding_dim must be specified together.")

    # Convert the last element to int, rest stay as str
    if args.train_augmented is not None:
        args.augment_techniques, args.augment_factor = parse_list_and_int(args.train_augmented)
    else:
        args.augment_techniques, args.augment_factor = None, None

    if args.hidden_dims is not None:
        args.hidden_dims = parse_list_ints(args.hidden_dims)
        args.model_args["hidden_dims"] = args.hidden_dims
    
    if args.out_channels is not None:
        args.out_channels = parse_list_ints(args.out_channels)
        args.model_args["out_channels"] = args.out_channels

    models = [args.one_svm, args.isolation_forest, args.local_outlier, args.autoencoder, args.anogan, args.cnn_anogan, args.cnn_supervised_1d, args.cnn_supervised_2d, args.lstm]
    if not any(models):
        parser.error("You must specify at least one of -o, -i, -l -a, --anogan, --cnn_anogan, --cnn_supervised_2d, --cnn_supervised_1d")

    try:
        manager = TrainingManager(args)
        manager.run()
    except Exception as e:
        log.exception("Unexpected failure")
        sys.exit(99)

if __name__ == "__main__":
    main()