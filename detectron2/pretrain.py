"""
Training of INoD Object Detection (Faster R-CNN) or Instance Segmentation (Mask R-CNN) based on detectron2
author: Julia Hindel
"""

import wandb
import os
import argparse
import torch

# detectron2
from detectron2.engine import launch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg

# local imports
from inod.train import INoDTrainer
from inod.predict import INoDPredictor
from inod.dataset.dataset_mapper import prep_source
from inod.utils import visualize_test_data_INoD

parser = argparse.ArgumentParser(description='INoD Training')
parser.add_argument('data_source', metavar='DIR',
                    help='path to dataset which is the source')
parser.add_argument('data_noise', metavar='DIR',
                    help='path to dataset which is used as noise')
parser.add_argument('challenge', help='challenge to load.', default='FP') # FP or SB16
parser.add_argument('--config-res', default='inod/configs/backbone/bn_backboneC5.yaml',
                    help='path to resnet cfg file')
parser.add_argument('--config_file', default='inod/configs/pretrain_sb16_fasterrcnn.yaml',
                    help='path to config file')
parser.add_argument('--output_dir', default='',
                    help='path to output directory')
parser.add_argument('--wandb', action='store_true',
                    default=False, help='track training on wandb.')
parser.add_argument('--test_name', default="dummy",
                    help='storage and test name on wandb')
parser.add_argument('--project_name', default="dummy",
                    help='project name on wandb')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use. For single GPU')
parser.add_argument('--crop_size', default=224, type=int,
                    help='size of crops from image')
parser.add_argument('--distributed', action='store_true',
                    default=False, help='distributed training')
parser.add_argument('--no_gpu', default=0, type=int,
                    help='No. of GPUs.')
parser.add_argument('--test', action='store_true',
                    default=False, help='Train or testing.')

# INoD settings
parser.add_argument('--random_noise', action='store_true',
                    default=False, help='inject noise from uniform distr.')
parser.add_argument('--noisy_ref', default=-1, type=int,
                    help='Noisy layer to use as ref size for noise injection.')
parser.add_argument('--noisy_layers', nargs="*",
                    default=["res2", "res3", "res4", "res5"], help='layers to inject noise.')
parser.add_argument('--noise_rate', default=0.2, type=float,
                    help='Proportion of noise to source features.')
parser.add_argument('--box_fill_rate', default=0.333333, type=float,
                    help='Number of filed elements in box.')
parser.add_argument('--noise_split', default="train",
                    help='split to use for noise')


def main(args):
    # set up detectron2 model cfg from command line args
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.OUTPUT_DIR = f"./output/{args.test_name}"
    cfg.DISTRIBUTED = args.distributed
    cfg.WANDB = args.wandb
    # set for single GPU training
    if not args.distributed:
        cfg.MODEL.DEVICE = args.gpu

    cfg.RANDOM_NOISE = args.random_noise
    cfg.DATA_SOURCE = args.data_source
    cfg.DATA_NOISE = args.data_noise
    cfg.CONFIG_RES = args.config_res
    cfg.NOISY_LAYERS = args.noisy_layers
    cfg.NOISY_REF = args.noisy_ref
    cfg.NOISE_RATE = args.noise_rate
    cfg.BOX_FILL_RATE = args.box_fill_rate
    cfg.NOISE_SPLIT = args.noise_split

    DatasetCatalog.clear()
    if args.challenge not in list(["SB16", "FP"]):
        print("CHALLENGE NOT SUPPORTED!")
        exit()

    if args.distributed:
        if torch.distributed.get_rank(group=None) == 0:
            if args.wandb:
                wandb.init(name=args.test_name, project=args.project_name, entity="X", sync_tensorboard=True,
                           settings=wandb.Settings(start_method="thread"))
            local_rank = 1
        else:
            local_rank = 0
    else:
        torch.multiprocessing.set_sharing_strategy('file_system')
        if args.wandb:
            wandb.init(name=args.test_name, project=args.project_name, entity="X", sync_tensorboard=True,
                       settings=wandb.Settings(start_method="thread"))
        local_rank = 1

    print(cfg)

    # detectron2 dataset init

    for d in ["train", "val"]:  # change here according to dataset
        DatasetCatalog.register(args.challenge + "_" + d, lambda path=args.data_source, split=d: prep_source(path, split))
        MetadataCatalog.get(args.challenge + "_" + d).set(thing_classes=["noise"])

    print("SETTING DATASET FINISHED")
    trainer = INoDTrainer(cfg)

    # prepare Trainer and train
    if not args.test:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer.resume_or_load(resume=True)
        trainer.train()

    # produce visualizations of training success
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = INoDPredictor(cfg)
    data_loader = trainer.build_test_loader(cfg, cfg.DATASETS.TRAIN[0])
    visualize_test_data_INoD(data_loader, predictor, cfg, MetadataCatalog, cfg.OUTPUT_DIR, mask=cfg.MODEL.MASK_ON)

    if local_rank == 1:
        with open(f"{cfg.OUTPUT_DIR}/config.yml", 'w') as f:
            f.write(cfg.dump())

        if args.wandb:
            # push to wandb
            wandb.save(f"{cfg.OUTPUT_DIR}/config.yml")
            wandb.save(f"{cfg.OUTPUT_DIR}/model_final.pth")

            wandb.finish()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.distributed:
        os.environ['MASTER_ADDR'] = "XXX"
        os.environ['MASTER_PORT'] = "X"
        launch(main, num_gpus_per_machine=args.no_gpu, num_machines=1, machine_rank=0, dist_url='env://', args=(args,),)
    else:
        main(parser.parse_args())
