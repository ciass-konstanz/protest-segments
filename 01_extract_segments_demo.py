#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Extract segments demo

"""

import os
import sys
import torch
import logging
import argparse
import requests
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from typing import NoReturn


def extract_segments_demo(
    args: argparse.Namespace,
    logger: logging.Logger
) -> NoReturn:
    """
    Extract segments demo

    :param args: arguments from parser (argparse.Namespace)
    :param logger: logger (logging.Logger)
    :return:
    """

    logger.info(f"Extract segments demo with arguments {args}")
    logger.info("Prepare segmenter")

    if not args.cpu:
        if args.vocabulary == "coco":
            # import modules only available with gpu
            sys.path.insert(0, "./modules/MaskDINO")

            from maskdino import add_maskformer2_config

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
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.0  # no effect
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # no effect
            cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.0 # no effect
            cfg.freeze()

            # predictor
            predictor = DefaultPredictor(cfg)

            # class names
            class_names = {class_num: class_name.lower() for class_num, class_name in enumerate(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)}
        elif args.vocabulary == "lvis":
            # import modules only available with gpu
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
            cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand" # load later
            cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
            cfg.freeze()
            # predictor
            predictor = LVISPredictor(cfg, args)
            # class names
            class_names = {class_num: class_name.lower() for class_num, class_name in enumerate(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)}

    else:
        if args.vocabulary == "coco":
            # config
            cfg = get_cfg()
            # device
            cfg.MODEL.DEVICE = "cpu"
            # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # set threshold for this model
            # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
            # predictor
            predictor = DefaultPredictor(cfg)
            # class names
            class_names = {class_num: class_name.lower() for class_num, class_name in enumerate(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)}
            print(cfg.DATASETS.TRAIN[0])
        elif args.vocabulary == "lvis":
            # config
            cfg = get_cfg()
            # device
            cfg.MODEL.DEVICE = "cpu"
            # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
            cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # set threshold for this model
            # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
            # predictor
            predictor = DefaultPredictor(cfg)
            # class names
            class_names = {class_num: class_name.lower() for class_num, class_name in enumerate(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)}
            print(cfg.DATASETS.TRAIN[0])
    # load image
    logger.info(f"Loading image {args.image}")
    img = np.array(Image.open(args.image).convert('RGB'))

    # run predictor on image
    logger.info("Running predictor")
    predictions = predictor(img)

    # visualize predictions
    logger.info("Visualizing predictions")
    visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.SEGMENTATION)
    visualized_output = visualizer.draw_instance_predictions(predictions["instances"].to("cpu"))
    visualized_output = Image.fromarray(visualized_output.get_image()[:, :, ::-1])
    visualized_output.save(args.image_segmented)

    # save segments
    logger.info("Saving segments")
    segments_long_df = pd.DataFrame([])

    for seg_num, seg_score, seg_mask in zip(predictions["instances"].pred_classes, predictions["instances"].scores, predictions["instances"].pred_masks):
        segments_long_df = pd.concat([segments_long_df, pd.Series({
            "id": "image_1",
            "seg_num": seg_num.item(),
            "seg_name": class_names[seg_num.item()],
            "seg_score": seg_score.item(),
            "seg_prop": (torch.sum(seg_mask == True) / torch.numel(seg_mask)).item()
        }).to_frame().T], ignore_index=True)

    if args.feature_aggregation == "bin":
        # get segments wide
        segments_wide_df = segments_long_df.pivot_table(index=["id"], columns=["seg_name"], values=["seg_score"], aggfunc={"seg_score": ["any"]})
        cols_missing = [category for category in class_names.values() if category not in [label for agg, var, label in segments_wide_df.columns]]
        cols_missing = [f'{args.vocabulary.lower()}_{category.lower()}_bin' for category in cols_missing]
        segments_wide_df.columns = [f'{args.vocabulary.lower()}_{label.lower()}_bin' for var, agg, label in segments_wide_df.columns]
        segments_wide_df = segments_wide_df.reset_index()
        segments_wide_df = segments_wide_df.reindex(columns=segments_wide_df.columns.tolist() + [column for column in cols_missing])
        segments_wide_df = segments_wide_df.fillna(0)
        # get cols
        cols = [f'{args.vocabulary.lower()}_{cat.lower()}_bin' for cat in class_names.values()]
        # get vector
        feature_vector = segments_wide_df.loc[:, cols].astype(int)

    elif args.feature_aggregation == "count":
        segments_wide_df = segments_long_df.pivot_table(index=["id"], columns=["seg_name"], values=["seg_score"], aggfunc={"seg_score": ["count"]})
        cols_missing = [category for category in class_names.values() if category not in [label for agg, var, label in segments_wide_df.columns]]
        cols_missing = [f'{args.vocabulary.lower()}_{category.lower()}_count' for category in cols_missing]
        segments_wide_df.columns = [f'{args.vocabulary.lower()}_{label.lower()}_count' for var, agg, label in segments_wide_df.columns]
        segments_wide_df = segments_wide_df.reset_index()
        segments_wide_df = segments_wide_df.reindex(columns=segments_wide_df.columns.tolist() + [column for column in cols_missing])
        segments_wide_df = segments_wide_df.fillna(0)
        # get cols
        cols = [f'{args.vocabulary.lower()}_{cat.lower()}_count' for cat in class_names.values()]
        # get vector
        feature_vector = segments_wide_df.loc[:, cols].astype(int)

    elif args.feature_aggregation == "area_max":
        segments_wide_df = segments_long_df.pivot_table(index=["id"], columns=["seg_name"], values=["seg_prop"], aggfunc={"seg_prop": ["max"]})
        cols_missing = [category for category in class_names.values() if category not in [label for agg, var, label in segments_wide_df.columns]]
        cols_missing = [f"{args.vocabulary.lower()}_{category.lower()}_area_max" for category in cols_missing]
        segments_wide_df.columns = [f"{args.vocabulary.lower()}_{label.lower()}_area_max" for var, agg, label in segments_wide_df.columns]
        segments_wide_df = segments_wide_df.reset_index()
        segments_wide_df = segments_wide_df.reindex(columns=segments_wide_df.columns.tolist() + [column for column in cols_missing])
        segments_wide_df = segments_wide_df.fillna(0)
        # get cols
        cols = [f"{args.vocabulary.lower()}_{cat.lower()}_area_max" for cat in class_names.values()]
        # get vector
        feature_vector = segments_wide_df.loc[:, cols].astype(float)

    elif args.feature_aggregation == "area_sum":
        segments_wide_df = segments_long_df.pivot_table(index=["id"], columns=["seg_name"], values=["seg_prop"], aggfunc={"seg_prop": ["sum"]})
        cols_missing = [category for category in class_names.values() if category not in [label for agg, var, label in segments_wide_df.columns]]
        cols_missing = [f"{args.vocabulary.lower()}_{category.lower()}_area_sum" for category in cols_missing]
        segments_wide_df.columns = [f"{args.vocabulary.lower()}_{label.lower()}_area_sum" for var, agg, label in segments_wide_df.columns]
        segments_wide_df = segments_wide_df.reset_index()
        segments_wide_df = segments_wide_df.reindex(columns=segments_wide_df.columns.tolist() + [column for column in cols_missing])
        segments_wide_df = segments_wide_df.fillna(0)
        # get cols
        cols = [f"{args.vocabulary.lower()}_{cat.lower()}_area_sum" for cat in class_names.values()]
        # get vector
        feature_vector = segments_wide_df.loc[:, cols].astype(float)

    # save feature vector
    feature_vector.to_csv(args.features, index=False)

    logger.info("Finished extracting segments demo")

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

    parser = argparse.ArgumentParser(description="Extract segments demo")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("docs/demo.jpg"),
        help="Path to image (default: docs/demo.jpg)"
    )
    parser.add_argument(
        "--vocabulary",
        type=str,
        choices=["coco", "lvis"],
        default="lvis",
        help="Name of vocabulary (default: lvis)",
    )
    parser.add_argument(
        "--feature-aggregation",
        type=str,
        choices=["bin", "count", "area_max", "area_sum"],
        default="count",
        help="Name of feature aggregation (default: count)",
    )
    parser.add_argument(
        "--image-segmented",
        type=Path,
        default=Path("docs/demo_segmented.jpg"),
        help="Path to segmented image (default: docs/demo_segmented.jpg)"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("docs/demo_features.csv"),
        help="Path to feature vector (default: docs/demo_features.csv)"
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

    # extract segments demo
    extract_segments_demo(args=args, logger=logger)
