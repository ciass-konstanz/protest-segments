#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Evaluate dataset

"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import NoReturn


def eval_dataset(
    args: argparse.Namespace,
    logger: logging.Logger
) -> NoReturn:
    """
    Evaluate dataset

    :param args: arguments from parser (argparse.Namespace)
    :param logger: logger (logging.Logger)
    :return:
    """

    logger.info(f"Starting evaluating dataset with arguments {args}")

    logger.info("Starting loading images")
    images = pd.read_csv(args.images, dtype={"protest": "Int64"})
    # code protest name and confidence
    images["protest_category"] = np.where(images["protest"].isin([2,3]), "Protest", "No Protest")
    images["protest_confidence"] = np.where(images["protest"].isin([0,3]), "High", "Low")

    # define countries
    countries = {
        "ARG": "Argentina", "BHR": "Bahrain", "CHL": "Chile", "DZA": "Algeria", "IDN":
        "Indonesia", "LBN": "Lebanon", "NGA": "Nigeria", "RUS": "Russia", "VEN":
        "Venezuela", "ZAF": "South Africa"
    }

    # create crosstab
    data = pd.crosstab(
        index=images.location,
        columns=[images["protest_category"], images["protest_confidence"]],
        margins=True,
        margins_name="Total"
    ).reindex(columns=pd.MultiIndex.from_tuples([("No Protest", "High"), ("No Protest", "Low"), ("Protest", "Low"), ("Protest", "High")], names=["", ""]))
    # rename columns and indexes
    data.rename(index=countries, inplace=True)
    data.index.set_names("", inplace=True)

    # convert table in latex
    table = data.style.format(thousands=",").to_latex(
        position="hb!",
        label="tab:data_full",
        caption="Images in protest images dataset",
        column_format="lrrrr",
        position_float="centering",
        hrules=True,
        multicol_align="c"
    )

    # save table
    with open(args.tables / "table_a2.tex", "w") as file:
        file.write(table)
        logger.info("Saved table a2")

    logger.info("Finished evaluating dataset")

    return


def get_logger() -> logging.Logger:
    """
    Get logger

    :return:
    """

    # logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # formatter
    formatter = logging.Formatter(fmt="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")

    # handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def get_parser() -> argparse.ArgumentParser:
    """
    Get parser

    :return: argument parser (argparse.ArgumentParser)
    """

    parser = argparse.ArgumentParser(description="Evaluate dataset")
    parser.add_argument(
        "--images",
        type=Path,
        help="Path to images"
    )
    parser.add_argument(
        "--tables",
        type=Path,
        help="Path to directory of tables"
    )

    return parser


if __name__ == "__main__":
    # logger
    logger = get_logger()

    # args
    args = get_parser().parse_args()

    # eval dataset
    eval_dataset(args=args, logger=logger)
