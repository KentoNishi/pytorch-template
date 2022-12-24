import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    type=str,
    required=True,
    help="Experiment config name.",
)
parser.add_argument(
    "-t",
    "--tag",
    type=str,
    required=True,
    help="Experiment tag name.",
)
cli_args = parser.parse_args()
config, tag = cli_args.config, f"{cli_args.config}-{cli_args.tag}"
