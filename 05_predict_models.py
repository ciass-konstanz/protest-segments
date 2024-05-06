#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Predict models

"""

import sys
import csv
import timm
import torch
import pickle
import random
import logging
import tarfile
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
from io import BytesIO
from pathlib import Path
from typing import NoReturn
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from detectron2.data import MetadataCatalog
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, get_worker_info


ImageFile.LOAD_TRUNCATED_IMAGES = True


class TarDataset(Dataset):
    """
    Tar dataset

    Source: https://github.com/jotaf98/simple-tar-dataset
    """

    def __init__(self, archive, images, transform=to_tensor, confidence="low", mode="all"):
        """
        Initialize tar dataset

        :param archive: path to Tar file containing the images (string or TarDataset)
        :param images: path fo csv file containing the images (string)
        :param transform: transformation pipeline, default: to_tensor (torchvision.transforms.functional)
        :param confidence: confidence level, low or high, default: low (str)
        :param mode: image from split, train, test or all, default: all (str)
        """

        if not isinstance(archive, TarDataset):
            worker = get_worker_info()
            worker = worker.id if worker else None
            self.tar_obj = {worker: tarfile.open(archive)}
            self.archive = archive

            # store headers of all files and folders by name
            members = sorted(self.tar_obj[worker].getmembers(), key=lambda m: m.name)
            self.members_by_name = {m.name: m for m in members}
        else:
            self.members_by_name = archive.members_by_name
            self.archive = archive.archive  # the original path to the Tar file
            self.tar_obj = {}  # will get filled by get_file on first access

        self.confidence = confidence
        self.mode = mode

        # assign a label to each image, based on text file
        self.ids = []
        self.samples = []
        self.targets = []

        with open(images) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not self.mode == "all" and self.mode != row["split"]:
                    continue
                # get target
                target = int(row["protest"])
                if self.confidence == "high":
                    if target == 0:
                        self.targets.append(0)
                    elif target == 3:
                        self.targets.append(1)
                    else:
                        continue
                elif self.confidence == "low":
                    if target == 0:
                        self.targets.append(0)
                    elif target == 1:
                        self.targets.append(0)
                    elif target == 2:
                        self.targets.append(1)
                    elif target == 3:
                        self.targets.append(1)
                    else:
                        continue
                else:
                    raise ValueError(f"Protest dataset got unknown confidence level {self.confidence}")
                # get id
                self.ids.append(row["id"])
                # get path
                self.samples.append(row["path"])

        self.transform = transform

    def __getitem__(self, index):
        """
        Return a single sample.
        """

        image = self.get_image(self.samples[index], pil=True)
        image = image.convert("RGB")  # if it's grayscale, convert to RGB
        if self.transform:  # apply any custom transforms
            image = self.transform(image)

        label = self.targets[index]

        return (image, label)

    def __len__(self):
        """
        Return the length of the dataset
        """

        return len(self.samples)

    def get_image(self, name, pil=False):
        """
        Read an image from the Tar archive
        """

        image = Image.open(BytesIO(self.get_file(name).read()))
        if pil:
          return image

        return to_tensor(image)

    def get_file(self, name):
        """
        Read an arbitrary file from the Tar archive
        """

        # ensure a unique file handle per worker, in multiprocessing settings
        worker = get_worker_info()
        worker = worker.id if worker else None

        if worker not in self.tar_obj:
          self.tar_obj[worker] = tarfile.open(self.archive)

        return self.tar_obj[worker].extractfile(self.members_by_name[name])

    def __del__(self):
        """
        Close the TarFile file handles on exit
        """

        for o in self.tar_obj.values():
          o.close()

    def __getstate__(self):
        """
        Serialize without the TarFile references, for multiprocessing compatibility
        """

        state = dict(self.__dict__)
        state['tar_obj'] = {}
        return state


class FinalLayer(nn.Module):
    """
    Modified last layer for Resnet by Won et al 2017

    Source: https://github.com/wondonghyeon/protest-detection-violence-estimation
    """

    def __init__(self):
        super(FinalLayer, self).__init__()
        self.fc = nn.Linear(2048, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out


def modified_resnet50():
    """
    Modify resnet50

    Source: https://github.com/wondonghyeon/protest-detection-violence-estimation
    """

    # load pretrained resnet50 with a modified last fully connected layer
    model = models.resnet50(pretrained=True)
    model.fc = FinalLayer()

    return model


def predict_models(
    args: argparse.Namespace,
    logger: logging.Logger
) -> NoReturn:
    """
    Predict models

    :param args: arguments from parser (argparse.Namespace)
    :param logger: logger (logging.Logger)
    :return:
    """

    logger.info(f"Starting predicting models with arguments {args}")

    # loop over methods
    if args.with_conventional:
        methods = ["logistic", "tree", "forest", "xgboost", "resnet", "resnet50", "vit"]
    else:
        methods = ["logistic", "tree", "forest", "xgboost"]

    for mod in methods:
        logger.info(f"Starting predicting models for classification method {mod}")
        if mod in methods[:4]:
            # images
            images_df = pd.read_csv(args.images)
            images_df = images_df.loc[(images_df["protest"].isin([0, 1, 2, 3]))]
            images_df["protest"] = images_df["protest"].replace({0: 0, 1: 0, 2: 1, 3: 1})

            # loop over vocabularies
            vocabularies = ["coco", "lvis"]
            for voc in vocabularies:
                logger.info(f"Starting predicting models for vocabulary {voc}")

                # loop over features
                features = ["bin", "count", "area_max", "area_sum"]
                for feat in features:
                    logger.info(f"Starting predicting models for feature {feat}")

                    if voc == "coco":
                        categories = [cat.lower() for cat in MetadataCatalog.get("coco_2017_train").thing_classes]
                        # segments
                        segments_df = pd.read_csv(args.segments_coco)
                        segments_df = segments_df.loc[(segments_df.id.isin(images_df.id) & (segments_df.seg_score >= 0.1))].reset_index(drop=True)
                    elif voc == "lvis":
                        categories = [cat.lower() for cat in MetadataCatalog.get("lvis_v1_train").thing_classes]
                        # segments
                        segments_df = pd.read_csv(args.segments_lvis)
                        segments_df = segments_df.loc[(segments_df.id.isin(images_df.id) & (segments_df.seg_score >= 0.1))].reset_index(drop=True)

                    if feat == "bin":
                        segments_df = segments_df.pivot_table(index=["id"], columns=["seg_name"], values=["seg_score"], aggfunc={"seg_score": ["any"]})
                        cols_missing = [category for category in categories if category not in [label for agg, var, label in segments_df.columns]]
                        cols_missing = [f"{voc}_{category}_seg_score_any" for category in cols_missing]
                        columns = [f"{voc}_{cat}_seg_score_any" for cat in categories]
                    elif feat == "count":
                        segments_df = segments_df.pivot_table(index=["id"], columns=["seg_name"], values=["seg_score"], aggfunc={"seg_score": ["count"]})
                        cols_missing = [category for category in categories if category not in [label for agg, var, label in segments_df.columns]]
                        cols_missing = [f"{voc}_{category}_seg_score_count" for category in cols_missing]
                        columns = [f"{voc}_{cat}_seg_score_count" for cat in categories]
                    elif feat == "area_max":
                        segments_df = segments_df.pivot_table(index=["id"], columns=["seg_name"], values=["seg_prop"], aggfunc={"seg_prop": ["max"]})
                        cols_missing = [category for category in categories if category not in [label for agg, var, label in segments_df.columns]]
                        cols_missing = [f"{voc}_{category}_seg_prop_max" for category in cols_missing]
                        columns = [f"{voc}_{cat}_seg_prop_max" for cat in categories]
                    elif feat == "area_sum":
                        segments_df = segments_df.pivot_table(index=["id"], columns=["seg_name"], values=["seg_prop"], aggfunc={"seg_prop": ["sum"]})
                        cols_missing = [category for category in categories if category not in [label for agg, var, label in segments_df.columns]]
                        cols_missing = [f"{voc}_{category}_seg_prop_sum" for category in cols_missing]
                        columns = [f"{voc}_{cat}_seg_prop_sum" for cat in categories]

                    segments_df = segments_df.sort_index(axis=1, level=1)
                    segments_df.columns = [f"{voc}_{label.lower()}_{var}_{agg}" for var, agg, label in segments_df.columns]
                    segments_df = segments_df.reset_index()
                    segments_df = segments_df.reindex(columns=segments_df.columns.tolist() + [column for column in cols_missing])

                    images_df = pd.merge(images_df, segments_df, how="left", on="id")
                    images_df = images_df.replace(to_replace={col: np.nan for col in images_df if col.startswith(f"{voc}_")}, value=0)
                    images_df = images_df.astype({col: int for col in images_df if col.endswith("_seg_score_any")})

                    # model
                    if mod in methods[:3]:
                        path = args.models / f"seg_low_{mod}_{voc}_{feat}.pkl"
                        with open(path, "rb") as file:
                            model = pickle.load(file)
                    elif mod == methods[3]:
                        # model
                        if args.cpu:
                            model = xgb.XGBClassifier(n_jobs=args.num_workers, importance_type="weight", random_state=args.seed)
                        else:
                            model = xgb.XGBClassifier(n_jobs=args.num_workers, importance_type="weight", gpu_id=0, predictor="gpu_predictor", tree_method="gpu_hist", random_state=args.seed)
                        path = args.models / f"seg_low_{mod}_{voc}_{feat}.pth"
                        model.load_model(path)

                    # scale features
                    if mod == methods[0]:
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaler.fit(images_df.loc[images_df["split"] == "train", columns])
                        images_df[columns] = scaler.transform(images_df[columns])

                    # outputs
                    outputs = []

                    batch_size = 10000
                    print_freq = 1
                    for count, start in enumerate(range(0, len(images_df), batch_size)):
                        if count % print_freq == 0:
                            logger.info(f"Starting predicting protest for image {start}")
                        output = model.predict(images_df.loc[start:start + batch_size - 1, columns])
                        outputs.append(output)

                    # save outputs
                    predictions_df = pd.DataFrame({
                        "id": images_df["id"],
                        "protest_pred": np.concatenate(outputs).ravel()
                    })
                    path = args.predictions / f"predictions_seg_low_{mod}_{voc}_{feat}.csv"
                    predictions_df.sort_values("id").to_csv(path, index=False, float_format="%f")

        elif mod == "resnet":
            # seed
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True

            # loader
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            dataset = TarDataset(archive=args.images_archive, images=args.images, transform=transform, confidence="low", mode="all")
            loader = DataLoader(dataset, batch_size=128, num_workers=args.num_workers, shuffle=False)

            # device
            device = torch.device("cpu" if args.cpu else "cuda:0")

            # model
            model = modified_resnet50()
            model = model.to(device)
            path = args.models / f"{mod}_low.pth"
            model.load_state_dict(torch.load(str(path))["state_dict"])

            outputs = torch.tensor([], device=device)

            # switch to evaluate mode
            model.eval()

            with torch.no_grad():
                for i, (images, _) in enumerate(loader):
                    if i % 100 == 0:
                        logger.info(f"Predicting image batch {i}")

                    # get images and target on device
                    images = images.to(device)
                    # get output
                    output = model(images)
                    # collect
                    outputs = torch.cat((outputs, output[:, 0]), 0)

            logger.info("Finished predicting image batches")

            # save outputs
            predictions_df = pd.DataFrame({
                "id": dataset.ids,
                "protest_pred": outputs.cpu().numpy()
            })
            path = args.predictions / f"predictions_{mod}_low.csv"
            predictions_df.sort_values("id").to_csv(path, index=False, float_format="%f")

        elif mod == "resnet50":
            # seed
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True

            # loader
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            dataset = TarDataset(archive=args.images_archive, images=args.images, transform=transform, confidence="low", mode="all")
            loader = DataLoader(dataset, batch_size=128, num_workers=args.num_workers, shuffle=False)

            # device
            device = torch.device("cpu" if args.cpu else "cuda:0")

            # model
            model = timm.create_model("resnet50", pretrained=False, num_classes=2)
            model = model.to(device)
            path = args.models / f"{mod}_low.pth"
            model.load_state_dict(torch.load(str(path)))

            outputs = torch.tensor([], device=device)

            # switch to evaluate mode
            model.eval()

            with torch.no_grad():
                for i, (images, _) in enumerate(loader):
                    if i % 100 == 0:
                        logger.info(f"Predicting image batch {i}")
                    # get images and target on device
                    images = images.to(device)
                    # get output
                    output = model(images)
                    # collect
                    outputs = torch.cat((outputs, torch.softmax(output, dim=1)[:, 1]), 0)

            logger.info("Finished predicting image batches")

            # save outputs
            predictions_df = pd.DataFrame({
                "id": dataset.ids,
                "protest_pred": outputs.cpu().numpy()
            })
            path = args.predictions / f"predictions_{mod}_low.csv"
            predictions_df.sort_values("id").to_csv(path, index=False, float_format="%f")

        elif mod == "vit":
            # seed
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True

            # loader
            transform = transforms.Compose([
                transforms.Resize(416),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            dataset = TarDataset(archive=args.images_archive, images=args.images, transform=transform, confidence="low", mode="all")
            loader = DataLoader(dataset, batch_size=128, num_workers=args.num_workers, shuffle=False)

            # device
            device = torch.device("cpu" if args.cpu else "cuda:0")

            # model
            model = timm.create_model("vit_base_patch16_384", pretrained=False, num_classes=2)
            model = model.to(device)
            path = args.models / f"{mod}_low.pth"
            model.load_state_dict(torch.load(str(path)))

            outputs = torch.tensor([], device=device)

            # switch to evaluate mode
            model.eval()

            with torch.no_grad():
                for i, (images, _) in enumerate(loader):
                    if i % 100 == 0:
                        logger.info(f"Predicting image batch {i}")
                    # get images and target on device
                    images = images.to(device)
                    # get output
                    output = model(images)
                    # collect
                    outputs = torch.cat((outputs, torch.softmax(output, dim=1)[:, 1]), 0)

            logger.info("Finished predicting image batches")

            # save outputs
            predictions_df = pd.DataFrame({
                "id": dataset.ids,
                "protest_pred": outputs.cpu().numpy()
            })
            path = args.predictions / f"predictions_{mod}_low.csv"
            predictions_df.sort_values("id").to_csv(path, index=False, float_format="%f")

    logger.info("Finished predicting protest")

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

    parser = argparse.ArgumentParser(description="Predict models")
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
        "--predictions",
        type=Path,
        help="Path to directory of predictions"
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
    parser.add_argument(
        "--with-conventional",
        action="store_true",
        help="Make predictions also with conventional methods"
    )

    return parser


if __name__ == "__main__":
    # logger
    logger = get_logger()

    # args
    args = get_parser().parse_args()

    # predict models
    predict_models(args=args, logger=logger)
