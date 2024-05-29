#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Evaluate temporal trends

"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import NoReturn
import matplotlib.dates as mdates


def eval_temporal(
    args: argparse.Namespace,
    logger: logging.Logger
) -> NoReturn:
    """
    Evaluate temporal trends

    :param args: arguments from parser (argparse.Namespace)
    :param logger: logger (logging.Logger)
    :return:
    """

    logger.info(f"Starting evaluating temporal trends with arguments {args}")

    logger.info("Starting loading images")
    images = pd.read_csv(args.images, dtype={"protest": "Int64"})
    # filter on protest images
    images = images.loc[images.protest.isin([2, 3])]
    # parse date
    images["date_dt"] = pd.to_datetime(images["date"], format="%Y-%m-%dT%H:%M:%SZ")

    logger.info("Starting loading segments")
    segments = pd.read_csv(args.segments_lvis)
    segments = segments.loc[(segments.id.isin(images.id) & (segments.seg_score >= 0.1))].reset_index(drop=True)
    segments = segments.pivot_table(index=["id"], columns=["seg_name"], values=["seg_score"], aggfunc={"seg_score": ["count"]})
    # change column names
    segments.columns = [f"lvis_{label}_{var}_{agg}" for var, agg, label in segments.columns]

    # merge images with segments
    images = pd.merge(images, segments, how="left", on="id")
    images = images.replace(to_replace={col: np.nan for col in images if col.startswith(f"lvis_")}, value=0)

    # define countries
    countries = {
        "ARG": "Argentina", "BHR": "Bahrain", "CHL": "Chile", "DZA": "Algeria", "IDN":
        "Indonesia", "LBN": "Lebanon", "NGA": "Nigeria", "RUS": "Russia", "VEN":
        "Venezuela", "ZAF": "South Africa"
    }

    # find most frequent segments per country
    features = images.groupby("location").agg({col: "sum" for col in segments.columns}).reset_index()
    features = pd.melt(features, id_vars="location", value_vars=segments.columns)

    # add feature names
    features["feature_name"] = features["variable"].replace({"_seg_score_count": ""}, regex=True).replace({"lvis_": ""}, regex=True).replace({"_": " "}, regex=True).str.title()
    # add country names
    features["country_name"] = features["location"].apply(lambda c: countries[c])

    # select three most frequent on protest images per country
    features = features.loc[features.groupby(["location"])["value"].nlargest(3).index.get_level_values(1)]

    sns.set_style("ticks")  # setting style
    sns.set_context("paper")  # setting context
    sns.set_palette("Greys_r")  # setting palette
    params = {"ytick.color": "black",
              "xtick.color": "black",
              "axes.labelcolor": "black",
              "axes.edgecolor": "black",
              #"text.usetex": True,
              #"font.family": "serif",
              #"font.serif": ["Computer Modern Serif"],
              "font.serif": ["Verdana"],
              "axes.labelsize": 9,
              "axes.titlesize": 9,
              "legend.fontsize": 7,
              "xtick.labelsize": 7,
              "ytick.labelsize": 7,
              }
    plt.rcParams.update(params)

    # create figure
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(6, 10))

    # iterate over features
    for row, (location, group1) in enumerate(features.groupby("location")):
        # get features
        feat = group1["variable"].tolist()
        # plot feature frequencies
        tmp = images.loc[images["location"]==location].set_index("date_dt").groupby(pd.Grouper(freq="D"))[feat].sum().reset_index()
        tmp = pd.wide_to_long(tmp, ["lvis"], i=["date_dt"], j="var", sep="_", suffix="\D+").reset_index()
        tmp["var"] = tmp["var"].replace({"_seg_score_count": ""}, regex=True).replace({"_": " "}, regex=True).str.title()
        sns.lineplot(data=tmp, x="date_dt", y="lvis", hue="var", ax=ax[row//2, row%2], palette="muted", errorbar=None)
        ax[row//2, row%2].set_xlabel(None)
        ax[row//2, row%2].set_ylabel(None)
        ax[row//2, row%2].set_yscale("log")
        ax[row//2, row%2].tick_params(which="minor", length=0)
        ax[row//2, row%2].set_title(countries[location])
        ax[row//2, row%2].legend(loc="upper right", edgecolor="none", facecolor="white", framealpha=0.8).set_title(None)
        ax[row//2, row%2].set_xticks(ax[row//2, row%2].get_xticks(), ax[row//2, row%2].get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        ax[row//2, row%2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    fig.tight_layout()
    plt.savefig(args.figures / "figure_a4.pdf", format="pdf", bbox_inches="tight", pad_inches = 0.01, dpi=200)
    logger.info("Saved figure a4")

    logger.info("Finished evaluating temporal trends")

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

    parser = argparse.ArgumentParser(description="Evaluate temporal trends")
    parser.add_argument(
        "--images",
        type=Path,
        help="Path to images"
    )
    parser.add_argument(
        "--segments-lvis",
        type=Path,
        help="Path to LVIS segments"
    )
    parser.add_argument(
        "--figures",
        type=Path,
        help="Path to directory of figures"
    )

    return parser


if __name__ == "__main__":
    # logger
    logger = get_logger()

    # args
    args = get_parser().parse_args()

    # eval temporal trends
    eval_temporal(args=args, logger=logger)
