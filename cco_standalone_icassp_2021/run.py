#!/usr/bin/env python3

import argparse
import importlib
import json
import os

from simulation import run


if __name__ == "__main__":
    """
    To run, pass the path to the JSON file as the first (only) parameter.
    """

    parser = argparse.ArgumentParser(description="CCO Simulation")
    parser.add_argument(
        "JSON_file",
        metavar="<JSON Parameter File>",
        type=str,
        help="<JSON parameter file>",
    )
    args = parser.parse_args()

    with open(os.path.abspath(args.JSON_file)) as f:
        json_data = json.load(f)
        run(json_data)
