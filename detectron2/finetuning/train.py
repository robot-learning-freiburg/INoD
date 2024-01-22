"""
Fine-tuning Trainer for detectron2 model.
author: Julia Hindel

"""

import os

# detectron2
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader

# local files
from finetuning.hooks.LossEvalHook import *
from finetuning.hooks.BNHook import *
from finetuning.dataset.dataset_mapper import DatasetMapperCustom

class TrainerFinetune(DefaultTrainer):
    """ Trainer  """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # set COCO as evaluator
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        # overwrite default train loader
        return build_detection_train_loader(cfg, mapper=DatasetMapperCustom(cfg.TYPE, True))

    @classmethod
    def build_test_loader(cls, cfg, name):
        # overwrite default test loader
        return build_detection_test_loader(cfg, name, mapper=DatasetMapperCustom(cfg.TYPE, False))

    def build_hooks(self):
        # loss evaluation hook
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0],
                                        mapper=DatasetMapperCustom(self.cfg.TYPE, False))))
        hooks.insert(-1, BNHook(
            self.model,
            build_detection_train_loader(self.cfg, mapper=DatasetMapperCustom(self.cfg.TYPE, True)), 400))
        return hooks

