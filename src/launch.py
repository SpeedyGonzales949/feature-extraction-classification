import argparse
import yaml
from scripts.train import train
from scripts.test import test
from scripts.visualization import visualize

import yaml
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def process_args(args):
    """
    Process the command line arguments and perform the corresponding action. Launch training, testing or visualising.

    Args:
        args (Namespace): The command line arguments.

    Returns:
        None
    """
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
        if args.test_only:
            test(config, args.weights)
        elif args.visualization_only:
            visualize(config, args.weights)
        else:
            train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file.")
    parser.add_argument(
        "--config", type=str, help="Path to the configuration file", required=True
    )
    parser.add_argument(
        "--visualization-only",
        action="store_true",
        help="Flag to indicate if only visualisation should be performed",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Flag to indicate if only testing should be performed",
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Path to the weights file for testing",
        default=None,
    )

    args = parser.parse_args()
    process_args(args)
