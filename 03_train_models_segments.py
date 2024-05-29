#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Train models segments

"""

import sys
import random
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from detectron2.data import MetadataCatalog
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import NoReturn


def train_models_segments(
    args: argparse.Namespace,
    logger: logging.Logger
) -> NoReturn:
    """
    Train models segments

    :param args: arguments from parser (argparse.Namespace)
    :param logger: logger (logging.Logger)
    :return:
    """

    logger.info(f"Starting training models segments with arguments {args}")

    # seed
    random.seed(args.seed)

    # images
    logger.info(f"Starting loading images")
    images_df = pd.read_csv(args.images)

    # categories
    categories = {}
    categories["coco"] = [cat.lower() for cat in MetadataCatalog.get("coco_2017_train").thing_classes]
    categories["lvis"] = [cat.lower() for cat in MetadataCatalog.get("lvis_v1_train").thing_classes]

    # coco segments
    logger.info(f"Starting loading segments")
    segments_coco_df = pd.read_csv(args.segments_coco)
    segments_coco_df = segments_coco_df.loc[(segments_coco_df["id"].isin(images_df["id"]) & (segments_coco_df["seg_score"] >= 0.1))].reset_index(drop=True)
    segments_coco_df = segments_coco_df.pivot_table(index=["id"], columns=["seg_name"], values=["seg_score", "seg_prop"], aggfunc={"seg_score": ["count", "any"], "seg_prop": ["max", "sum"]})
    cols_missing = [category for category in categories["coco"] if category not in [label for agg, var, label in segments_coco_df.columns]]
    cols_missing = [f"coco_{category}_seg_score_count" for category in cols_missing] + [f"coco_{category}_seg_score_any" for category in cols_missing] + [f"coco_{category}_seg_prop_max" for category in cols_missing] + [f"coco_{category}_seg_prop_sum" for category in cols_missing]
    segments_coco_df = segments_coco_df.sort_index(axis=1, level=1)
    segments_coco_df.columns = [f"coco_{label}_{var}_{agg}" for var, agg, label in segments_coco_df.columns]
    segments_coco_df = segments_coco_df.reset_index()
    segments_coco_df = segments_coco_df.reindex(columns=segments_coco_df.columns.tolist() + [column for column in cols_missing])

    # lvis segments
    segments_lvis_df = pd.read_csv(args.segments_lvis)
    segments_lvis_df = segments_lvis_df.loc[(segments_lvis_df["id"].isin(images_df["id"]) & (segments_lvis_df["seg_score"] >= 0.1))].reset_index(drop=True)
    segments_lvis_df = segments_lvis_df.pivot_table(index=["id"], columns=["seg_name"], values=["seg_score", "seg_prop"], aggfunc={"seg_score": ["count", "any"], "seg_prop": ["max", "sum"]})
    cols_missing = [category for category in categories["lvis"] if category not in [label for agg, var, label in segments_lvis_df.columns]]
    cols_missing = [f"lvis_{category}_seg_score_count" for category in cols_missing] + [f"lvis_{category}_seg_score_any" for category in cols_missing] + [f"lvis_{category}_seg_prop_max" for category in cols_missing] + [f"lvis_{category}_seg_prop_sum" for category in cols_missing]
    segments_lvis_df = segments_lvis_df.sort_index(axis=1, level=1)
    segments_lvis_df.columns = [f"lvis_{label}_{var}_{agg}" for var, agg, label in segments_lvis_df.columns]
    segments_lvis_df = segments_lvis_df.reset_index()
    segments_lvis_df = segments_lvis_df.reindex(columns=segments_lvis_df.columns.tolist() + [column for column in cols_missing])

    # loop over confidence levels
    confidences = ["low", "high"]
    for conf in confidences:
        logger.info(f"Starting creating models for confidence level {conf}")

        # filter images by confidence level
        if conf == "low":
            images_conf_df = images_df.loc[(images_df["protest"].isin([0, 1, 2, 3]))]
            images_conf_df.loc[:, "protest"] = images_conf_df["protest"].replace({0: 0, 1: 0, 2: 1, 3: 1})
        elif conf == "high":
            images_conf_df = images_df.loc[(images_df["protest"].isin([0, 3]))]
            images_conf_df.loc[:, "protest"] = images_conf_df["protest"].replace({0: 0, 3: 1})

        # merge segments with images
        images_conf_df = pd.merge(images_conf_df, segments_coco_df, how="left", on="id")
        images_conf_df = pd.merge(images_conf_df, segments_lvis_df, how="left", on="id")
        images_conf_df = images_conf_df.replace(to_replace={col: np.nan for col in images_conf_df if col.startswith("coco_") or col.startswith("lvis_")}, value=0)
        images_conf_df = images_conf_df.astype({col: int for col in images_conf_df if col.endswith("_seg_score_any")})

        # loop over classification methods
        methods = ["logistic", "tree", "forest", "xgboost"]
        for mod in methods:
            logger.info(f"Starting creating models for classification method {mod}")

            if mod == "logistic":
                # model
                model = LogisticRegression(n_jobs=args.num_workers, random_state=args.seed)
                # params
                params = {
                    "solver": ["saga"],
                    "max_iter": [1000],
                    "penalty": ["l2"],
                    "C" : [0.01, 0.1, 1, 10]
                }
            elif mod == "tree":
                # model
                model = DecisionTreeClassifier(random_state=args.seed)
                # params
                params = {
                    "max_depth": [1, 2, 4, 8, 16]
                }
            elif mod == "forest":
                # model
                model = RandomForestClassifier(n_jobs=args.num_workers, random_state=args.seed)
                # params
                params = {
                    "n_estimators": [1, 10, 100, 1000, 2000, 10000],
                    "max_features": ["sqrt", "log2"],
                    "max_depth": [1, 2, 4, 8, 16],
                    "min_samples_leaf": [1, 2]
                }
            elif mod == "xgboost":
                # model
                if args.cpu:
                    model = xgb.XGBClassifier(n_jobs=args.num_workers, importance_type="weight")
                else:
                    model = xgb.XGBClassifier(n_jobs=args.num_workers, importance_type="weight", gpu_id=0, predictor="gpu_predictor", tree_method="gpu_hist")
                # params
                params = {
                    "colsample_bytree": [0.8],
                    "gamma": [0.001, 0.01],
                    "learning_rate": [0.1],
                    "max_depth": [1, 2, 4, 8, 16],
                    "n_estimators": [1, 10, 100, 1000, 2000, 10000],
                    "subsample": [0.8]
                }

            # loop over vocabularies
            vocabularies = ["coco", "lvis"]
            for voc in vocabularies:
                # loop over features
                features = ["bin", "count", "area_max", "area_sum"]
                for feat in features:
                    if feat == "bin":
                        cols = [f"{voc}_{cat}_seg_score_any" for cat in categories[voc]]
                    elif feat == "count":
                        cols = [f"{voc}_{cat}_seg_score_count" for cat in categories[voc]]
                    elif feat == "area_max":
                        cols = [f"{voc}_{cat}_seg_prop_max" for cat in categories[voc]]
                    elif feat == "area_sum":
                        cols = [f"{voc}_{cat}_seg_prop_sum" for cat in categories[voc]]

                    # get training set
                    X_train = images_conf_df.loc[(images_conf_df.split == "train"), cols]
                    y_train = images_conf_df.loc[(images_conf_df.split == "train"), "protest"]

                    # scale data for logistic
                    if mod == "logistic":
                        scaler = MinMaxScaler(feature_range = (0,1))
                        scaler.fit(X_train)
                        X_train = scaler.transform(X_train)

                    # gridsearch
                    logger.info(f"Starting training model for confidence {conf}, classification method {mod}, vocabulary {voc} and feature {feat}")
                    search = GridSearchCV(
                        model,
                        param_grid=params,
                        scoring=["accuracy", "precision", "recall", "roc_auc", "f1"],
                        refit="f1",
                        cv=5,
                        verbose=1,
                        error_score=np.nan,
                        return_train_score=True
                    )
                    search.fit(X_train, y_train)
                    logger.info(f"Finished training model with scores {search.best_score_} and parameters {search.best_params_}")

                    # save model
                    if mod == "xgboost":
                        path = args.models / f"seg_{conf}_{mod}_{voc}_{feat}.pth"
                        search.best_estimator_.save_model(path)
                    else:
                        path = args.models / f"seg_{conf}_{mod}_{voc}_{feat}.pkl"
                        with open(path, "wb") as f:
                            pickle.dump(search.best_estimator_, f)
                    logger.info(f"Saved model {path}")

    logger.info("Finished training models segments")

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

    parser = argparse.ArgumentParser(description="Train models segments")
    parser.add_argument(
        "--images",
        type=Path,
        help="Path to images"
    )
    parser.add_argument(
        "--segments-coco",
        type=Path,
        help="Path to COCO segments"
    )
    parser.add_argument(
        "--segments-lvis",
        type=Path,
        help="Path to LVIS segments"
    )
    parser.add_argument(
        "--models",
        type=Path,
        help="Path to directory of models"
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
        action='store_true',
        help="Use CPU only"
    )

    return parser


if __name__ == "__main__":
    # logger
    logger = get_logger()

    # args
    args = get_parser().parse_args()

    # train protest
    train_models_segments(args=args, logger=logger)
