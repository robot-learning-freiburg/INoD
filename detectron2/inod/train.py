"""
Trainer for INoD Pre-training.
author: Julia Hindel, largely adopted from detectron2: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved. (https://github.com/facebookresearch/detectron2)
"""

import os

# detectron2
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader, build_detection_test_loader

# local imports
from . import DatasetMapperINoD, LossEvalHook, COCOEvaluatorINoD


class INoDTrainer(DefaultTrainer):
    """ Trainer for INoD detection model"""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # set custom COCO evaluator
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluatorINoD(dataset_name, output_dir=output_folder, allow_cached_coco=False)

    @classmethod
    def build_train_loader(cls, cfg):
        # overwrite default train loader
        return build_detection_train_loader(cfg, mapper=DatasetMapperINoD(cfg))

    @classmethod
    def build_test_loader(cls, cfg, name):
        # overwrite default test loader
        return build_detection_test_loader(cfg, name, mapper=DatasetMapperINoD(cfg))

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(self.cfg, self.cfg.DATASETS.TRAIN[0], mapper=DatasetMapperINoD(self.cfg))))
        return hooks
