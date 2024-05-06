#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Evaluate models

"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import NoReturn


def eval_models(
    args: argparse.Namespace,
    logger: logging.Logger
) -> NoReturn:
    """
    Train models segments

    :param args: arguments from parser (argparse.Namespace)
    :param logger: logger (logging.Logger)
    :return:
    """

    logger.info(f"Starting evaluating models with arguments {args}")

    # read images
    images = pd.read_csv(args.images)
    # recode protest labels
    images["protest"] = images["protest"].replace({0: 0, 1: 0, 2: 1, 3: 1})

    # read models
    models_df = pd.read_csv(args.models)
    models_df = models_df.set_index("name")

    for model_name, model in models_df.iterrows():
        logger.info(f"Starting evaluating model {model_name}")
        # get predictions
        predictions = pd.read_csv(args.predictions / model["predictions"])
        # merge predictions with images
        predictions = pd.merge(images, predictions.loc[:, ["id", "protest_pred"]], how="left", on="id")
        # evaluate model on training set
        y_train_pred = predictions.loc[predictions.split == "train", "protest_pred"]
        y_train_pred = np.where(y_train_pred >= 0.5, 1, 0)
        y_train = predictions.loc[predictions.split == "train", "protest"].to_numpy(dtype=np.int64)
        # precision
        prec_train = precision_score(y_train, y_train_pred)
        # recall
        rec_train = recall_score(y_train, y_train_pred)
        # f1
        f1_train = f1_score(y_train, y_train_pred)

        # evaluate model on testing set
        y_test_pred = predictions.loc[predictions.split == "test", "protest_pred"]
        y_test_pred = np.where(y_test_pred >= 0.5, 1, 0)
        y_test = predictions.loc[predictions.split == "test", "protest"].to_numpy(dtype=np.int64)
        # precision
        prec_test = precision_score(y_test, y_test_pred)
        # recall
        rec_test = recall_score(y_test, y_test_pred)
        # f1
        f1_test = f1_score(y_test, y_test_pred)
        # save results
        models_df.loc[model_name, ["train_prec", "train_rec", "train_f1", "test_prec", "test_rec", "test_f1"]] = prec_train, rec_train, f1_train, prec_test, rec_test, f1_test

    # prepare table
    table_df = models_df.drop(columns=["approach", "vocabulary", "feature", "method", "weights", "predictions"])

    # change columns
    table_df.columns = pd.MultiIndex.from_tuples(
        [
            ("train", "train_prec"), ("train", "train_rec"), ("train", "train_f1"),
            ("test", "test_prec"), ("test", "test_rec"), ("test", "test_f1")
        ], names=["", ""]
    )

    # rename columns
    table_df = table_df.rename(columns={"train": "Training", "test": "Testing"},level=0).rename(
        columns={"train_prec": "Precision", "test_prec": "Precision", "train_rec": "Recall",
                 "test_rec": "Recall", "train_f1": "F1", "test_f1": "F1"}, level=1
    )

    # convert table in latex
    table = table_df.style.format(thousands=",", precision=4).to_latex(
        position="ht",
        label="table:models",
        caption=("Evaluation of different methods", "Evaluation of different methods"),
        column_format="lrrrrrr",
        position_float="raggedleft",
        hrules=True,
        multicol_align="c"
    )

    # save table
    with open(args.tables / "table_a3.tex", "w") as file:
        file.write(table)

    # set up figure params
    sns.set_style("ticks")
    sns.set_context("paper")
    sns.set_palette("Greys_r")
    params = {"ytick.color": "black",
              "xtick.color": "black",
              "axes.labelcolor": "black",
              "axes.edgecolor": "black",
              "text.usetex": True,
              "font.family": "serif",
              "font.serif": ["Computer Modern Serif"],
              "axes.labelsize": 9,
              "axes.titlesize": 9,
              "legend.fontsize": 7,
              "xtick.labelsize": 7,
              "ytick.labelsize": 7,
    }
    plt.rcParams.update(params)

    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=False, figsize=(5.85, 3.76))

    # plot segments (w/ coco)
    tmp = models_df.loc[(models_df["approach"] == "seg") & (models_df["vocabulary"] == "coco")]
    sns.barplot(data=tmp, x="feature", y="test_f1", hue="method", ax=ax[0], palette="Greys_r")

    ax[0].set_title("Segments (COCO)")
    ax[0].tick_params(axis="x", which="major", bottom=False, pad=0)
    ax[0].set_xlabel(None)
    ax[0].set_ylabel("F1 score")
    ax[0].set_ylim([0.0, 1.0])
    ax[0].set_box_aspect(1)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha="right", fontsize=7)

    ax[0].legend(handles=[tuple(bar_group) for bar_group in ax[0].containers],
                 labels=["Logistic", "Tree", "Forest", "Boosted tree"],
                 facecolor="white", edgecolor="white", framealpha=1,
                 bbox_to_anchor=(1.0, 1.0), loc="upper right", borderpad=0, fontsize=6,
                 frameon=False, markerfirst=False, handletextpad=0.3, handlelength=0.8,
                 ncol=2, columnspacing=0.8)

    # plot segments (w/ lvis)
    tmp = models_df.loc[(models_df["approach"] == "seg") & (models_df["vocabulary"] == "lvis")]
    sns.barplot(data=tmp, x="feature", y="test_f1", hue="method", ax=ax[1], palette="Greys_r")

    ax[1].set_title("Segments (LVIS)")
    ax[1].tick_params(axis="x", which="major", bottom=False, pad=0)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel(None)
    ax[1].set_ylim([0.0, 1.0])
    ax[1].get_legend().set_visible(False)
    ax[1].set_box_aspect(1)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax[1].set_yticklabels([], fontsize=9)

    ax[1].legend(handles=[tuple(bar_group) for bar_group in ax[1].containers],
                 labels=["Logistic", "Tree", "Forest", "Boosted tree"],
                 facecolor="white", edgecolor="white", framealpha=1,
                 bbox_to_anchor=(1.0, 1.0), loc="upper right", borderpad=0, fontsize=6,
                 frameon=False, markerfirst=False, handletextpad=0.3, handlelength=0.8,
                 ncol=2, columnspacing=0.8)

    # plot computer vision
    tmp = models_df.loc[(models_df["approach"] == "deep")]

    sns.barplot(data=tmp, x="approach", y="test_f1", hue="method", ax=ax[2], palette="Greys_r", width=0.3)

    ax[2].set_title("Traditional methods")
    ax[2].tick_params(axis="x", which="major", bottom=False, pad=0)
    ax[2].set_xlabel(None)
    ax[2].set_ylabel(None)
    ax[2].set_ylim([0.0, 1.0])
    ax[2].set_box_aspect(1)
    ax[2].set_xticklabels([], fontsize=7)
    ax[2].set_yticklabels([], fontsize=7)

    ax[2].legend(handles=[tuple(bar_group) for bar_group in ax[2].containers],
                 labels=["ResNet (Won et al., 2017)", "ResNet", "ViT"],
                 facecolor="white", edgecolor="white", framealpha=1,
                 bbox_to_anchor=(1.0, 1.0), loc="upper right", borderpad=0, fontsize=6,
                 frameon=False, markerfirst=False, handletextpad=0.3, handlelength=0.8)

    # save figure
    plt.savefig(args.figures / "figure_4.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=200)

    logger.info(f"Finished evaluating models")

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

    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument(
        "--images",
        type=Path,
        help="Path to images"
    )
    parser.add_argument(
        "--models",
        type=Path,
        help="Path to models"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Path to directory of predictions"
    )
    parser.add_argument(
        "--tables",
        type=Path,
        help="Path to directory of tables"
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

    # eval models
    eval_models(args=args, logger=logger)
