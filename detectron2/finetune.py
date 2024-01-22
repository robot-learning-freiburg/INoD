"""
Fine-tune model with detectron2.
author: Julia Hindel, largely adopted from detectron2: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved. (https://github.com/facebookresearch/detectron2)
"""

import argparse
import wandb
import os
import pickle
import random
import numpy as np
import torch

# detectron2
from detectron2.engine import DefaultPredictor, launch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# imports from files in same directory
from finetuning.train import TrainerFinetune
from finetuning.dataset.dataset import prep_SB16, prep_FP22
from finetuning.utils import visualize_test_data_wandb

parser = argparse.ArgumentParser(description='INoD Fine-tuning')
parser.add_argument('--weights', metavar='DIR', default=None,
                    help='pre-training weights')
parser.add_argument('--challenge', help='challenge to load', default='FP')  # FP or SB16
parser.add_argument('--config', default='finetuning/configs/finetune_fp_fasterrcnn.yaml',
                    help='path to config file')
parser.add_argument('--path', metavar='DIR', default="../fp_label",
                    help='path to dataset')
parser.add_argument('--data_split', default="ImageSets",
                    help='path to txt files of dataset.')
parser.add_argument('--test_name', default="dummy",
                    help='storage and test name on wandb')
parser.add_argument('--norm_path', default=None,
                    help='path to mean and std files of dataset')
parser.add_argument('--wandb', action='store_true',
                    default=False, help='track training on wandb.')
parser.add_argument('--project_name', default="dummy",
                    help='project name on wandb')
parser.add_argument('--BGR', action='store_true',
                    default=False, help='if images are loaded in RGB or BGR.')
parser.add_argument('--distributed', action='store_true',
                    default=False, help='distributed training')
parser.add_argument('--no_gpu', default=0, type=int,
                    help='No. of GPUs.')
parser.add_argument('--test', action='store_true',
                    default=False, help='Train or testing.')
parser.add_argument('--visualization', action='store_true',
                    default=False, help='To enable visualization.')


def init(args):
    """
    initialize training run (build detectron2 cfg, register datasets)
    :param args:
    :return:
    """

    # set seeds
    seed_val = 42
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # load config file (examples are saved in configs folder)
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.DATASETS.TEST_FINAL = (args.challenge + "_test",)
    cfg.TYPE = 'RGB' if not args.BGR else 'BGR'
    cfg.OUTPUT_DIR = f"./output/{args.test_name}"

    # create custom DatasetCatalog (detectron2 annotation format)
    DatasetCatalog.clear()

    if args.challenge == "SB16":
        prep = prep_SB16
        thing_classes = thing_classes_prep = ["crop", "weed"]
    elif args.challenge == "FP":
        prep = prep_FP22
        thing_classes = thing_classes_prep = ["potato", "weed"]
    else:
        print("CHALLENGE NOT SUPPORTED!")
        exit()

    for d in ["train", "val", "test"]:
        DatasetCatalog.register(args.challenge + "_" + d, lambda path=args.path,
                                                                 d=d,
                                                                 dataset_split=args.data_split,
                                                                 mask=cfg.MODEL.MASK_ON,
                                                                 thing_classes_prep=thing_classes_prep: prep(path,
                                                                                                             dataset_split,
                                                                                                             d, mask,
                                                                                                             thing_classes_prep))
        MetadataCatalog.get(args.challenge + "_" + d).set(thing_classes=thing_classes)

    if args.norm_path is not None:
        # load pre-computed mean and std
        mean_file = os.path.join(args.norm_path, 'mean')
        with open(mean_file, "rb") as fp:
            mean = pickle.load(fp)
        mean = mean.tolist()[:3]

        std_file = os.path.join(args.norm_path, 'std')
        with open(std_file, "rb") as fp:
            std = pickle.load(fp)
        std = std.tolist()[:3]

        cfg.MODEL.PIXEL_MEAN = mean
        cfg.MODEL.PIXEL_STD = std

    # adapt RESNET to pre-trained model setting
    if args.weights:
        if ("byol" in args.weights) or ("densecl" in args.weights) \
                or ("moco" in args.weights) or ("insloc" in args.weights) or \
                ("torch" in args.weights) or ("barlow" in args.weights):
            cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False

    return cfg


def main(args):
    cfg = init(args)

    if args.distributed:
        local_rank = 0
        if torch.distributed.get_rank(group=None) == 0:
            local_rank = 1
    else:
        local_rank = 1

    if local_rank == 1:
        if args.wandb:
            wandb.init(name=args.test_name, project=args.project_name, entity="X", sync_tensorboard=True,
                       settings=wandb.Settings(start_method="thread"))

    trainer = TrainerFinetune(cfg)

    if not args.test:
        # make output file and train
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer.resume_or_load(resume=False)
        trainer.train()

    print("FINAL TEST")
    # path to the trained model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(args.challenge + "_test", output_dir=cfg.OUTPUT_DIR)
    test_loader = trainer.build_test_loader(cfg, cfg.DATASETS.TEST_FINAL)
    results = inference_on_dataset(predictor.model, test_loader, evaluator)

    if local_rank == 1:
        if args.wandb:
            wandb.log({"test": results})

            # visualize testing images with prediction
            if args.visualization:
                visualize_test_data_wandb(test_loader, predictor, cfg, MetadataCatalog)

        with open(f"{cfg.OUTPUT_DIR}/config.yml", 'w') as f:
            f.write(cfg.dump())

        # save everything to wandb
        if args.wandb:
            wandb.save(f"{cfg.OUTPUT_DIR}/coco_instances_results.json")
            wandb.save(f"{cfg.OUTPUT_DIR}/config.yml")
            wandb.finish()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.distributed:
        os.environ['MASTER_ADDR'] = "XXX"
        os.environ['MASTER_PORT'] = "X"
        launch(main, num_gpus_per_machine=args.no_gpu, num_machines=1, machine_rank=0, dist_url='env://',
               args=(args,), )
    else:
        main(parser.parse_args())
