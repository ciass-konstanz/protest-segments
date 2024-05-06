#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Train models conventional

"""

import sys
import csv
import timm
import torch
import random
import logging
import tarfile
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.backends.cudnn as cudnn
from io import BytesIO
from PIL import Image, ImageFile
from torchvision import transforms
from pathlib import Path
from skorch import NeuralNetClassifier
from skorch.helper import SliceDataset
from skorch.callbacks import LRScheduler, EarlyStopping, PrintLog
from sklearn.model_selection import GridSearchCV
from torchvision.transforms.functional import to_tensor
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, get_worker_info
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import NoReturn


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


def train_models_conventional(
    args: argparse.Namespace,
    logger: logging.Logger
) -> NoReturn:
    """
    Train models conventional

    :param args: arguments from parser (argparse.Namespace)
    :param logger: logger (logging.Logger)
    :return:
    """

    logger.info(f"Starting training models conventional with arguments {args}")

    # seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    # results
    results = pd.DataFrame()

    # loop over methods
    methods = ["resnet50", "vit"]
    for mod in methods:
        logger.info(f"Starting creating models for classification method {mod}")

        if mod == "resnet50":
            # transform pipeline
            transform = transforms.Compose([
                transforms.RandomResizedCrop(256),  # 416
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.CenterCrop(224),  # 384
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            # device
            device = torch.device("cuda:0")
            # model
            model = timm.create_model("resnet50", pretrained=True, num_classes=2)
            model = model.to(device)
            net = NeuralNetClassifier(
                model,
                criterion=nn.CrossEntropyLoss,
                optimizer=torch.optim.SGD,
                max_epochs=100,
                iterator_train__num_workers=args.num_workers,
                iterator_valid__num_workers=args.num_workers,
                batch_size=128,
                optimizer__lr=0.001,
                device=device,
                train_split=False,
                verbose=False,
                callbacks=[
                    ("lr_scheduler", LRScheduler(policy=ReduceLROnPlateau, factor=0.1, patience=10)),
                    ("early_stopping", EarlyStopping(monitor="train_loss", patience=10, threshold=0.0001, sink=logger.info)),
                    ("logger", PrintLog(sink=logger.info))
                ],
            )
            # hyperparams
            params = [{"optimizer__lr": [1e-03, 1e-04], "optimizer__momentum": [0.99, 0.9]}]

        elif mod == "vit":
            # transform pipeline
            transform = transforms.Compose([
                transforms.RandomResizedCrop(416),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            # device
            device = torch.device("cuda:0")
            # model
            model = timm.create_model("vit_base_patch16_384", pretrained=True, num_classes=2)
            model = model.to(device)
            net = NeuralNetClassifier(
                model,
                criterion=nn.CrossEntropyLoss,
                optimizer=torch.optim.SGD,
                max_epochs=100,
                iterator_train__num_workers=args.num_workers,
                iterator_valid__num_workers=args.num_workers,
                batch_size=32,
                optimizer__lr=0.001,
                device=device,
                train_split=False,
                verbose=False,
                callbacks=[
                    ("lr_scheduler", LRScheduler(policy=ReduceLROnPlateau, factor=0.1, patience=10)),
                    ("early_stopping", EarlyStopping(monitor="train_loss", patience=10, threshold=0.0001, sink=logger.info)),
                    ("logger", PrintLog(sink=logger.info))
                ],
            )
            # hyperparams
            params = [{"optimizer__lr": [1e-06, 1e-05], "optimizer__momentum": [0.99, 0.9]}]

        # datasets
        dataset = TarDataset(archive=args.images_archive, images=args.images, transform=transform, confidence="low", mode="train")
        dataset_sliced = SliceDataset(dataset=dataset)
        dataset_y = np.array(dataset.targets)

        train_indices, val_indices = train_test_split(
            list(range(len(dataset.targets))),
            test_size=0.2,
            shuffle=True,
            random_state=args.seed
        )

        # create grid search
        search = GridSearchCV(
            net,
            param_grid=params,
            scoring=["accuracy", "precision", "recall", "roc_auc", "f1"],
            refit="f1",
            cv=iter([(train_indices, val_indices)]),
            verbose=3,
            error_score="raise",
            return_train_score=True,
        )
        search.fit(dataset_sliced, dataset_y)
        logger.info(f"Finished training model with scores {search.best_score_} and parameters {search.best_params_}")

        # store results
        results_iter = pd.DataFrame(search.cv_results_)
        results_iter["conf"] = "low"
        results_iter["method"] = mod

        # append results
        results = pd.concat([results, results_iter], axis=0, ignore_index=True)
        results.to_csv(args.trainings, index=False, float_format="%f")

        # save model
        path = args.models / f"{mod}_low.pth"
        search.best_estimator_.save_params(f_params=str(path))
        logger.info(f"Saved model {path}")

    logger.info("Finished training models conventional")

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

    parser = argparse.ArgumentParser(description="Train models conventional")
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
        "--trainings",
        type=Path,
        help="Path to trainings"
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

    return parser


if __name__ == "__main__":
    # logger
    logger = get_logger()

    # args
    args = get_parser().parse_args()

    # train models
    train_models_conventional(args=args, logger=logger)
