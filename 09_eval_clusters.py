#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Evaluate clusters

"""

import csv
import sys
import timm
import torch
import logging
import tarfile
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import NoReturn
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def crop_image(image):
    """
    Crop image

    :param image_in:
    :param image_out:
    :return:
    """

    width  = image.size[0]
    height = image.size[1]
    aspect = width / float(height)

    ideal_width = 500
    ideal_height = 500

    ideal_aspect = ideal_width / float(ideal_height)

    if aspect > ideal_aspect:
        # crop the left and right edges
        new_width = int(ideal_aspect * height)
        offset = (width - new_width) / 2
        resize = (offset, 0, width - offset, height)
    else:
        # crop the top and bottom
        new_height = int(width / ideal_aspect)
        offset = (height - new_height) / 2
        resize = (0, offset, width, height - offset)

    thumb = image.crop(resize).resize((ideal_width, ideal_height), Image.ANTIALIAS)

    return thumb


def eval_clusters(
    args: argparse.Namespace,
    logger: logging.Logger
) -> NoReturn:
    """
    Evaluate clusters

    :param args: arguments from parser (argparse.Namespace)
    :param logger: logger (logging.Logger)
    :return:
    """

    logger.info(f"Starting evaluating clusters with arguments {args}")

    # create embeddings
    logger.info("Starting creating embeddings")

    # device
    device = torch.device("cpu" if args.cpu else "cuda:0")

    transform = transforms.Compose([
        transforms.Resize(416),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # model
    model = timm.create_model("vit_base_patch16_384", pretrained=False, num_classes=2, exportable=True)
    model = model.to(device)
    checkpoint = torch.load(str(args.models / "vit_low.pth"))
    model.load_state_dict(checkpoint)
    model = create_feature_extractor(model, return_nodes=["pre_logits"])
    model.eval()

    # open images and image archive
    with open(args.images, mode="r", encoding="utf8") as images_file, tarfile.open(args.images_archive, mode="r") as images_archive_file:
        # set up dataset text file
        images_reader = csv.DictReader(images_file)
        with open(args.embeddings / "embeddings.csv", mode="w", encoding="utf8", buffering=1048576) as embeddings_file:
            # set up encodings file
            embeddings_fieldnames = ["id"]
            embeddings_fieldnames.extend([f"dim{x}" for x in range(768)])
            embeddings_writer = csv.DictWriter(embeddings_file, fieldnames=embeddings_fieldnames)
            embeddings_writer.writeheader()
            # loop over images in image file
            for image_num, image in enumerate(images_reader):
                # track progress
                if image_num % 10000 == 0:
                    logger.info(f"Getting embeddings from image {image_num}")
                # read image
                img = Image.open(BytesIO(images_archive_file.extractfile(image["path"]).read()))
                img = img.convert("RGB")
                img = transform(img)
                img = img.unsqueeze_(0)
                img = img.to(device)
                # run model on image to get encodings
                embeddings = model(img)["pre_logits"].cpu().flatten().detach().numpy()
                # prepare encodings
                embeddings = {f"dim{dim}": f"{value:.8f}" for dim, value in enumerate(np.nditer(embeddings, order='C'))}
                embeddings["id"] = image["id"]
                # write encodings
                embeddings_writer.writerow(embeddings)

    logger.info("Saved embeddings")

    # load embeddings
    embeddings = pd.read_csv(args.embeddings / "embeddings.csv")
    cols = [f"dim{x}" for x in range(768)]
    embeddings.loc[:, cols] = StandardScaler().fit_transform(embeddings.loc[:, cols])

    # cluster embeddings
    kmeans = KMeans(n_clusters=30, init="k-means++", n_init=1, random_state=args.seed).fit(embeddings.loc[:, cols])

    # get distances
    dist = np.take_along_axis(kmeans.transform(embeddings.loc[:, cols]), kmeans.labels_[:,None], axis=1).squeeze() ** 2

    # save clusters
    clusters = pd.DataFrame({"image_id": embeddings["id"], "cluster_id": kmeans.labels_, "distance": dist})
    clusters.sort_values(["cluster_id", "image_id"]).to_csv(args.clusters / "clusters.csv", index=False, float_format="%f")
    logger.info("Saved clusters")

    # read images
    images = pd.read_csv(args.images)
    # recode protest labels
    images["protest"] = images["protest"].replace({0: 0, 1: 0, 2: 1, 3: 1})

    # merge images with clusters
    images = pd.merge(images, clusters, left_on="id", right_on="image_id", how="left")

    # create dataframe
    index = list(range(30))
    columns = ["cluster_name", "TN", "FN", "TP", "FP", "Precision", "Recall", "F1"]
    models_cluster_df = pd.DataFrame(index=index, columns=columns)

    # read predictions
    predictions = pd.read_csv(args.predictions / "predictions_seg_low_xgboost_lvis_area_sum.csv")

    # merge predictions with images
    predictions = pd.merge(images, predictions.loc[:, ["id", "protest_pred"]], how="left", on="id")

    # group by location
    for cluster_id, cluster_predictions in predictions.groupby("cluster_id"):
        # evaluate model on training set
        y_train_pred = cluster_predictions.loc[cluster_predictions.split == "train", "protest_pred"]
        y_train_pred = np.where(y_train_pred >= 0.5, 1, 0)
        y_train = cluster_predictions.loc[cluster_predictions.split == "train", "protest"].to_numpy(dtype=np.int64)
        # precision
        prec_train = precision_score(y_train, y_train_pred, zero_division=np.nan)
        #models_cluster_df.loc[cluster_id, "Precision"] = prec_train
        # recall
        rec_train = recall_score(y_train, y_train_pred, zero_division=np.nan)
        #models_cluster_df.loc[cluster_id, "Recall"] = rec_train
        # f1
        f1_train = f1_score(y_train, y_train_pred, zero_division=np.nan)
        #models_cluster_df.loc[cluster_id, "F1"] = f1_train
        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred, labels=[0,1]).ravel()
        #models_cluster_df.loc[cluster_id, "TN"] = tn
        #models_cluster_df.loc[cluster_id, "FP"] = fp
        #models_cluster_df.loc[cluster_id, "FN"] = fn
        #models_cluster_df.loc[cluster_id, "TP"] = tp

        # evaluate model on testing set
        y_test_pred = cluster_predictions.loc[cluster_predictions.split == "test", "protest_pred"]
        y_test_pred = np.where(y_test_pred >= 0.5, 1, 0)
        y_test = cluster_predictions.loc[cluster_predictions.split == "test", "protest"].to_numpy(dtype=np.int64)
        # precision
        prec_test = precision_score(y_test, y_test_pred, zero_division=np.nan)
        models_cluster_df.loc[cluster_id, "Precision"] = prec_test
        # recall
        rec_test = recall_score(y_test, y_test_pred, zero_division=np.nan)
        models_cluster_df.loc[cluster_id, "Recall"] = rec_test
        # f1
        f1_test = f1_score(y_test, y_test_pred, zero_division=np.nan)
        models_cluster_df.loc[cluster_id, "F1"] = f1_test
        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0,1]).ravel()
        models_cluster_df.loc[cluster_id, "TN"] = tn
        models_cluster_df.loc[cluster_id, "FP"] = fp
        models_cluster_df.loc[cluster_id, "FN"] = fn
        models_cluster_df.loc[cluster_id, "TP"] = tp

    # define cluster names
    cluster_names = {
        0: "Landscapes with sky", 1: "Random", 2: "Texts", 3: "People in public", 4:
        "Texts", 5: "Football", 6: "Screenshots", 7: "Africans", 8: "Edited images", 9:
        "Streets", 10: "Football", 11: "Random", 12: "Selfies", 13:
        "Protest with flags", 14: "Woman", 15: "Gathering", 16: "Fire smoke", 17:
        "African gatherings", 18: "Concerts", 19: "Comics", 20: "Screenshots", 21:
        "Large mass protests", 22: "State police", 23: "Protest with signboards", 24:
        "Africans", 25: "Texts", 26: "Flags", 27: "Suit", 28: "Headpiece", 29:
        "Random images with text"
    }

    models_cluster_df["cluster_name"] = models_cluster_df.index.to_series().apply(lambda cluster_id: cluster_names[cluster_id]).values

    # increase index by 1
    models_cluster_df.index = models_cluster_df.index + 1

    # convert table in latex
    table = models_cluster_df.loc[models_cluster_df["TP"] + models_cluster_df["FN"] >= 20].fillna("").style.format(thousands=",", precision=4).to_latex(
        position="hb!",
        label="tab:model_cluster",
        caption="Evaluation of classifier per cluster",
        column_format="llrrrrrrr",
        position_float="centering",
        hrules=True,
        multicol_align="c"
    )

    # save table
    with open(args.tables / "table_a5.tex", "w") as file:
        file.write(table)

    logger.info("Saved table a5")

    # filter images
    clust = models_cluster_df.loc[models_cluster_df["TP"] + models_cluster_df["FN"] >= 20].index.tolist()
    clust = [c-1 for c in clust]
    clust_names = {id-1: cluster["cluster_name"] for id, cluster in models_cluster_df.iterrows()}

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

    # create plot
    fig, ax = plt.subplots(nrows=len(clust), ncols=3, figsize=(4, 8))

    # open image
    with tarfile.open(args.images_archive, mode="r") as images_archive_file:
        # visualize clusters
        for cluster_num, (cluster_id, cluster) in enumerate(images.loc[images["cluster_id"].isin(clust)].groupby("cluster_id")):
            # get quantiles
            cluster["quantile"] = pd.qcut(images["distance"], q=[0, .33, .67, 1.], labels=[0, 1, 2])
            # get first image
            img1 = cluster.loc[cluster["quantile"]==0].sample(n=1, random_state=12).squeeze()
            img1 = Image.open(BytesIO(images_archive_file.extractfile(img1["path"]).read()))
            img1 = crop_image(img1)
            ax[cluster_num, 0].imshow(img1)
            ax[cluster_num, 0].axis("off")
            ax[cluster_num, 0].set_xlabel(None)
            ax[cluster_num, 0].set_xticks([])
            ax[cluster_num, 0].set_ylabel(None)
            ax[cluster_num, 0].set_yticks([])
            ax[cluster_num, 0].annotate(f'{cluster_id+1}', (-3.25, 0.5), xycoords = 'axes fraction', rotation = 0, va = 'center') # fontweight = 'bold',
            ax[cluster_num, 0].annotate(f'{clust_names[cluster_id]}', (-2.75, 0.5), xycoords = 'axes fraction', rotation = 0, va = 'center')
            # get second image
            img2 = cluster.loc[cluster["quantile"]==1].sample(n=1, random_state=7).squeeze()
            img2 = Image.open(BytesIO(images_archive_file.extractfile(img2["path"]).read()))
            img2 = crop_image(img2)
            ax[cluster_num, 1].imshow(img2)
            ax[cluster_num, 1].axis("off")
            ax[cluster_num, 1].set_xlabel(None)
            ax[cluster_num, 1].set_xticks([])
            ax[cluster_num, 1].set_ylabel(None)
            ax[cluster_num, 1].set_yticks([])
            #ax[cluster_num, 1].set_title(cluster_names[cluster_id], pad=3)
            # get third image
            img3 = cluster.loc[cluster["quantile"]==2].sample(n=1, random_state=4).squeeze()
            img3 = Image.open(BytesIO(images_archive_file.extractfile(img3["path"]).read()))
            img3 = crop_image(img3)
            ax[cluster_num, 2].imshow(img3)
            ax[cluster_num, 2].axis('off')
            ax[cluster_num, 2].set_xlabel(None)
            ax[cluster_num, 2].set_xticks([])
            ax[cluster_num, 2].set_ylabel(None)
            ax[cluster_num, 2].set_yticks([])

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.1)
    plt.savefig(args.figures / "figure_a1.pdf", format="pdf", bbox_inches="tight", pad_inches = 0.01, dpi=200)
    logger.info("Saved figure a1")

    logger.info("Finished evaluating clusters")

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

    parser = argparse.ArgumentParser(description="Evaluate clusters")
    parser.add_argument(
        "--images",
        type=Path,
        help="Path to images"
    )
    parser.add_argument(
        "--images-archive",
        type=Path,
        help="Path to archive of images"
    )
    parser.add_argument(
        "--models",
        type=Path,
        help="Path to directory of models"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Path to directory of predictions"
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Path to directory of embeddings"
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        help="Path to directory of clusters"
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
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

    # eval clusters
    eval_clusters(args=args, logger=logger)
