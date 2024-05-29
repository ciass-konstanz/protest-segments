#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Extract segments

"""

import os
import csv
import sys
import torch
import logging
import tarfile
import requests
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from typing import NoReturn

sys.path.insert(0, "./modules/MaskDINO")

from maskdino import add_maskformer2_config

sys.path.insert(0, "./modules/Detic")
sys.path.insert(0, "./modules/Detic/third_party/CenterNet2")

from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder


class LVISPredictor(DefaultPredictor):
    def __init__(self, cfg, args, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """

        self.BUILDIN_CLASSIFIER = {
            "lvis": "modules/Detic/datasets/metadata/lvis_v1_clip_a+cname.npy",
            "objects365": "modules/Detic/datasets/metadata/o365_clip_a+cnamefix.npy",
            "openimages": "modules/Detic/datasets/metadata/oid_clip_a+cname.npy",
            "coco": "modules/Detic/datasets/metadata/coco_clip_a+cname.npy",
        }

        self.BUILDIN_METADATA_PATH = {
            "lvis": "lvis_v1_val",
            "objects365": "objects365_v2_val",
            "openimages": "oid_val_expanded",
            "coco": "coco_2017_val",
        }

        if args.vocabulary == "custom":
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(',')
            classifier = get_clip_embeddings(self.metadata.thing_classes)
        else:
            self.metadata = MetadataCatalog.get(
                self.BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = self.BUILDIN_CLASSIFIER[args.vocabulary]

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = DefaultPredictor(cfg)
        reset_cls_test(self.predictor.model, classifier, num_classes)

    def __call__(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
        """

        predictions = self.predictor(image)

        return predictions


def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


def extract_segments(
    args: argparse.Namespace,
    logger: logging.Logger
) -> NoReturn:
    """
    Extract segments

    :param args: arguments from parser (argparse.Namespace)
    :param logger: logger (logging.Logger)
    :return:
    """

    logger.info(f"Extract segments with arguments {args}")
    logger.info("Prepare segmenter")

    if args.vocabulary == "coco":
        # download model weights
        model_path = "models/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth"
        model_url = "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth"

        if not os.path.exists(model_path):
            logger.info("Download model weights")
            r = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(r.content)

        # config
        cfg = get_cfg()
        # for poly lr schedule
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file("modules/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml")
        cfg.MODEL.WEIGHTS = "models/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth"
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold # no effect
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold # no effect
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold # no effect
        cfg.freeze()
        # predictor
        predictor = DefaultPredictor(cfg)
        # class names
        class_names = {class_num: class_name.lower() for class_num, class_name in enumerate(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)}
    elif args.vocabulary == "lvis":
        # download model weights
        model_path = "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        model_url = "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

        if not os.path.exists(model_path):
            logger.info("Download model weights")
            r = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(r.content)

        # config
        cfg = get_cfg() # weight-file
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file("modules/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        cfg.MODEL.WEIGHTS = "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = "modules/Detic/datasets/metadata/lvis_v1_train_cat_info.json"
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand" # load later
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        cfg.freeze()
        # predictor
        predictor = LVISPredictor(cfg, args)
        # class names
        class_names = {class_num: class_name.lower() for class_num, class_name in enumerate(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)}

    logger.info(f"Prepare dataset")

    # open image file and archive
    with tarfile.open(args.images_archive, mode="r") as images_archive_file, open(args.images, mode="r", encoding="utf8") as images_file:
        # set up dataset text file
        images_reader = csv.DictReader(images_file)
        with open(args.segments, mode="w", encoding="utf8", buffering=1048576) as segments_file:
            # set up segments file
            segments_fieldnames = ["id", "seg_num", "seg_name", "seg_score", "seg_prop"]
            segments_writer = csv.DictWriter(segments_file, fieldnames=segments_fieldnames)
            segments_writer.writeheader()
            # loop over images in image file
            for image_num, image in enumerate(images_reader):
                # track progress
                if image_num % 1000 == 0:
                    logger.info(f"Getting segments from image {image_num}")
                # read image
                img = Image.open(BytesIO(images_archive_file.extractfile(image["path"]).read()))
                img = img.convert("RGB")
                img = np.asarray(img)
                img = img[:, :, ::-1]
                # run predictor on image
                predictions = predictor(img)
                # write segments
                for seg_num, seg_score, seg_mask in zip(predictions["instances"].pred_classes, predictions["instances"].scores, predictions["instances"].pred_masks):
                    if seg_score >= args.confidence_threshold:
                        segments_writer.writerow({
                            "id": image["id"],
                            "seg_num": seg_num.item(),
                            "seg_name": class_names[seg_num.item()],
                            "seg_score": f"{seg_score.item():.8f}",
                            "seg_prop": f"{(torch.sum(seg_mask == True) / torch.numel(seg_mask)).item():.8f}"
                        })

    logger.info("Finished extracting segments")

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

    parser = argparse.ArgumentParser(description="Extract segments")
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
        "--segments",
        type=Path,
        help="Path to segments"
    )
    parser.add_argument(
        "--vocabulary",
        type=str,
        choices=["coco", "lvis"],
        help="Name of vocabulary",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown (default: 0.1)",
    )

    return parser


if __name__ == "__main__":
    # logger
    logger = get_logger()

    # args
    args = get_parser().parse_args()

    # extract segments
    extract_segments(args=args, logger=logger)
