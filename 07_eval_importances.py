#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Evaluate importances

"""

import sys
import random
import logging
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from detectron2.data import MetadataCatalog
from sklearn.inspection import permutation_importance
from typing import NoReturn


def eval_importances(
    args: argparse.Namespace,
    logger: logging.Logger
) -> NoReturn:
    """
    Evaluate importances

    :param args: arguments from parser (argparse.Namespace)
    :param logger: logger (logging.Logger)
    :return:
    """

    logger.info(f"Starting evaluating segments with arguments {args}")

    # seed
    random.seed(args.seed)

    # images
    logger.info("Starting loading images")
    images_df = pd.read_csv(args.images)
    # recode protest labels
    images_df["protest"] = images_df["protest"].replace({0: 0, 1: 0, 2: 1, 3: 1})

    # categories
    categories = [cat.lower() for cat in MetadataCatalog.get("lvis_v1_train").thing_classes]

    # segments
    logger.info("Starting loading segments")
    segments_df = pd.read_csv(args.segments_lvis)
    segments_df = segments_df.loc[(segments_df.id.isin(images_df["id"]) & (segments_df["seg_score"] >= 0.1))].reset_index(drop=True)

    # features
    segments_df = segments_df.pivot_table(index=["id"], columns=["seg_name"], values=["seg_prop"], aggfunc={"seg_prop": ["sum"]})
    cols_missing = [category for category in categories if category not in [label.lower() for agg, var, label in segments_df.columns]]
    cols_missing = [f"lvis_{category.lower()}_seg_prop_sum" for category in cols_missing]
    cols = [f"lvis_{cat.lower()}_seg_prop_sum" for cat in categories]

    segments_df = segments_df.sort_index(axis=1, level=1)
    segments_df.columns = [f"lvis_{label.lower()}_{var}_{agg}" for var, agg, label in segments_df.columns]
    segments_df = segments_df.reset_index()
    segments_df = segments_df.reindex(columns=segments_df.columns.tolist() + [column for column in cols_missing])

    images_df = pd.merge(images_df, segments_df, how="left", on="id")
    images_df = images_df.replace(to_replace={col: np.nan for col in images_df if col.startswith(f"lvis_")}, value=0)
    images_df = images_df.astype({col: int for col in images_df if col.endswith("_seg_score_any")})

    # model
    logger.info(f"Starting loading model")
    if args.cpu:
        model = xgb.XGBClassifier(n_jobs=args.num_workers, importance_type="weight", random_state=args.seed)
    else:
        model = xgb.XGBClassifier(n_jobs=args.num_workers, importance_type="weight", gpu_id=0, predictor="gpu_predictor", tree_method="gpu_hist", random_state=args.seed)

    model.load_model(args.models / "seg_low_xgboost_lvis_area_sum.pth")

    # features
    features_df = pd.DataFrame()

    for location, group in images_df.groupby("location"):
        # feature importances
        logger.info(f"Starting getting feature importances for location {location}")
        importances = permutation_importance(
            model,
            group.loc[group["split"] == "train", cols],
            group.loc[group["split"] == "train", "protest"],
            scoring=["precision", "recall", "f1"],
            n_repeats=5,
            n_jobs=1,
            random_state=args.seed
        )

        importances_df = pd.DataFrame({
            "feature": cols,
            "location": location,
            "imp_prec_mean": importances["precision"].importances_mean.flatten(),
            "imp_prec_std": importances["precision"].importances_std.flatten(),
            "imp_rec_mean": importances["recall"].importances_mean.flatten(),
            "imp_rec_std": importances["recall"].importances_std.flatten(),
            "imp_f1_mean": importances["f1"].importances_mean.flatten(),
            "imp_f1_std": importances["f1"].importances_std.flatten()
        })

        # feature counts
        logger.info(f"Starting getting areas for location {location}")
        props_df = group.loc[group["split"] == "train"].groupby(["protest"])[cols].mean().reset_index()
        props_df = pd.melt(props_df, id_vars="protest", value_vars=cols)
        props_df = pd.DataFrame([{
                "feature": seg_name,
                "area_sum_protest": props_df.loc[(props_df.variable == seg_name) & (props_df.protest == 1.0), "value"].to_list()[0],
                "area_sum_noprotest": props_df.loc[(props_df.variable == seg_name) & (props_df.protest == 0.0), "value"].to_list()[0],
                "area_sum_diff": props_df.loc[(props_df.variable == seg_name) & (props_df.protest == 1.0), "value"].to_list()[0] - props_df.loc[(props_df.variable == seg_name) & (props_df.protest == 0.0), "value"].to_list()[0]
        } for seg_name in props_df.variable.unique()]
        )
        # features
        features_location_df = pd.merge(importances_df, props_df, on="feature", how="left")
        features_df = pd.concat([features_df, features_location_df], ignore_index=True)

    # save features
    features_df.to_csv(args.features / "features_seg_low_xgboost_lvis_area_sum.csv", index=False, float_format="%f")

    # create dict for countries
    countries = {
        "ARG": "Argentina", "BHR": "Bahrain", "CHL": "Chile", "DZA": "Algeria",
        "IDN": "Indonesia", "LBN": "Lebanon", "NGA": "Nigeria", "RUS": "Russia",
        "VEN": "Venezuela", "ZAF": "South Africa"
    }

    # add feature names
    features_df["feature_name"] = features_df["feature"].replace({"_seg_prop_sum": ""}, regex=True).replace({"lvis_": ""}, regex=True).replace({"_": " "}, regex=True).str.capitalize()
    # add country names
    features_df["location_name"] = features_df["location"].apply(lambda c: countries[c])

    # find features with largest area sum in protest images
    features_area_sum_most = features_df.groupby(["feature_name"])[["imp_f1_mean", "area_sum_protest", "area_sum_noprotest"]].mean().reset_index().nlargest(10, columns="area_sum_protest")["feature_name"].to_list()

    # find features with large importances
    features_area_sum_important = features_df.groupby(["feature_name"])[["imp_f1_mean", "area_sum_protest", "area_sum_noprotest"]].mean().reset_index().nlargest(10, columns="imp_f1_mean")["feature_name"].to_list()

    # style figure
    sns.set_style("ticks")  # setting style
    sns.set_context("paper")  # setting context
    sns.set_palette("Greys_r")  # setting palette
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

    # create figure
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(5.85, 2.26))

    # plot feature frequencies
    tmp = features_df.loc[features_df["feature_name"].isin(features_area_sum_most)]
    tmp = pd.wide_to_long(tmp, ["area_sum"], i=["feature_name", "location"], j="protest", sep="_", suffix="\D+").reset_index()
    tmp = tmp.loc[tmp.protest.isin(["protest", "noprotest"])].replace({"protest": "Protest", "noprotest": "No Protest"})
    sns.barplot(data=tmp, x="feature_name", y="area_sum", hue="protest", order=features_area_sum_most, orient="v", ax=ax[0], palette="Greys_r", errorbar=None)  # errorbar="sd", errwidth=2,

    ax[0].tick_params(axis="x", which="major", bottom=False, pad=0)
    ax[0].set_xlabel(None)
    ax[0].set_ylabel("Proportion")
    ax[0].set_ylim([0.0, 0.4])
    ax[0].get_legend().set_visible(False)
    ax[0].set_box_aspect(1)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax[0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax[0].set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=8)

    ax[0].legend(handles=[tuple(bar_group) for bar_group in ax[0].containers],
                 labels=["Protest", "No protest"],
                 facecolor="white", edgecolor="white", framealpha=1,
                 bbox_to_anchor=(1.0, 1.0), loc="upper right", borderpad=0, fontsize=8,
                 frameon=False, markerfirst=False, handletextpad=0.3, handlelength=0.8)

    # plot feature importances
    tmp = features_df.loc[features_df["feature_name"].isin(features_area_sum_important)]
    sns.barplot(data=tmp, x="feature_name", y="imp_f1_mean", order=features_area_sum_important, orient="v", ax=ax[1], color=(0.40784313725490196, 0.40784313725490196, 0.40784313725490196, 1.0), errorbar=None)

    ax[1].tick_params(axis="x", which="major", bottom=False, pad=0)
    ax[1].set_xlabel(None)
    ax[1].set_ylabel("F1 score")
    ax[1].set_ylim([0.0, 0.23])
    ax[1].set_box_aspect(1)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax[1].set_yticks([0.0, 0.05, 0.1, 0.15, 0.2])
    ax[1].set_yticklabels([0.0, 0.05, 0.1, 0.15, 0.2], fontsize=8)

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(args.figures / "figure_5.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=200)
    logger.info("Saved figure with similar importances")

    # find features with large deviations
    tmp1 = features_df.groupby(["feature_name"])[["imp_f1_mean", "area_sum_protest", "area_sum_noprotest"]].agg({"imp_f1_mean": ["mean", "std"]}).reset_index()
    tmp1.columns = ["_".join(a) for a in tmp1.columns.to_flat_index()]
    # calculate distance to mean importance of all countries
    tmp2 = features_df.copy(deep=True)
    tmp2["imp_f1_mean_mean"] = tmp2.apply(lambda row: tmp1.loc[(tmp1.feature_name_ == row["feature_name"]), "imp_f1_mean_mean"].tolist()[0], axis=1)
    tmp2["imp_f1_mean_dist"] = tmp2.apply(lambda row: row["imp_f1_mean"] - row["imp_f1_mean_mean"], axis=1)
    tmp2["imp_f1_mean_rdist"] = tmp2["imp_f1_mean_dist"] / tmp2["imp_f1_mean_mean"]
    tmp2["imp_f1_mean_dist_abs"] = tmp2["imp_f1_mean_dist"].abs()
    # list features
    #tmp2.loc[tmp2["feature_name"].isin(features_area_sum_important)].nlargest(20, columns="imp_f1_mean_rdist")[["location", "feature_name", "imp_f1_mean_dist", "imp_f1_mean_rdist"]]

    # get importances for all countries for poster
    tmp = tmp2.loc[tmp2["feature_name"] == "Poster"].sort_values("imp_f1_mean_dist", ascending=False)
    tmp["imp_f1_mean_norm"] = (tmp["imp_f1_mean_dist"] - tmp["imp_f1_mean_dist"].mean()) / tmp["imp_f1_mean_dist"].std()

    # create figure
    f, ax = plt.subplots(figsize=(2.92, 3.76))
    sns.set_palette("deep")
    ax = sns.barplot(data=tmp, x="imp_f1_mean_norm", y="location_name", ax=ax, color=(0.40784313725490196, 0.40784313725490196, 0.40784313725490196, 1.0))

    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticks([], minor=True)
    ax.tick_params(axis="y", which="major", left=False)
    ax.set_box_aspect(1)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    l = tmp["imp_f1_mean_norm"].abs().max() * 1.1
    plt.xlim([-l, l])

    # save figure
    plt.savefig(args.figures / "figure_6_1.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=200)
    logger.info("Saved figure with different importances for posters")

    # get importances for all countries for car
    tmp = tmp2.loc[tmp2["feature_name"] == "Car (automobile)"].sort_values("imp_f1_mean_dist", ascending=False)
    tmp["imp_f1_mean_norm"] = (tmp["imp_f1_mean_dist"] - tmp["imp_f1_mean_dist"].mean()) / tmp["imp_f1_mean_dist"].std()

    # create figure
    f, ax = plt.subplots(figsize=(2.92, 3.76))
    sns.set_palette("deep")
    ax = sns.barplot(data=tmp, x="imp_f1_mean_norm", y="location_name", ax=ax, color=(0.40784313725490196, 0.40784313725490196, 0.40784313725490196, 1.0))

    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticks([], minor=True)
    ax.tick_params(axis="y", which="major", left=False)
    ax.set_box_aspect(1)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    l = tmp["imp_f1_mean_norm"].abs().max() * 1.1
    plt.xlim([-l, l])

    # save figure
    plt.savefig(args.figures / "figure_6_2.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=200)
    logger.info("Saved figure with different importances for car")

    # get importances for all countries for candle
    tmp = tmp2.loc[tmp2["feature_name"] == "Candle"].sort_values("imp_f1_mean_dist", ascending=False)
    tmp["imp_f1_mean_norm"] = (tmp["imp_f1_mean_dist"] - tmp["imp_f1_mean_dist"].mean()) / tmp["imp_f1_mean_dist"].std()

    # create figure
    f, ax = plt.subplots(figsize=(2.92, 3.76))
    sns.set_palette("deep")
    ax = sns.barplot(data=tmp, x="imp_f1_mean_norm", y="location_name", ax=ax, color=(0.40784313725490196, 0.40784313725490196, 0.40784313725490196, 1.0))

    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticks([], minor=True)
    ax.tick_params(axis="y", which="major", left=False)
    ax.set_box_aspect(1)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    l = tmp["imp_f1_mean_norm"].abs().max() * 1.1
    plt.xlim([-l, l])

    # save figure
    plt.savefig(args.figures / "figure_6_3.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=200)
    logger.info("Saved figure with different importances for candle")

    logger.info("Finished evaluating importances")

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

    parser = argparse.ArgumentParser(description="Evaluate importances")
    parser.add_argument(
        "--images",
        type=Path,
        help="Path to images"
    )
    parser.add_argument(
        "--segments-lvis",
        type=Path,
        help="Path to lvis segments"
    )
    parser.add_argument(
        "--models",
        type=Path,
        help="Path to directory of models"
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Path to directory of features"
    )
    parser.add_argument(
        "--figures",
        type=Path,
        help="Path to directory of figures"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers (default: 8)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU only"
    )

    return parser


if __name__ == "__main__":
    # logger
    logger = get_logger()

    # args
    args = get_parser().parse_args()

    # eval importances
    eval_importances(args=args, logger=logger)
