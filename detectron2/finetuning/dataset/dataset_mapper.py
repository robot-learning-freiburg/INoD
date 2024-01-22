"""
Custom DatasetMapper to overwrite detectron2 default version.
author: Julia Hindel, largely adopted from detectron2: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved. (https://github.com/facebookresearch/detectron2)
"""


import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils

import copy
import random
import torch

class DatasetMapperCustom():
    def __init__(self, input_format, training):
        super().__init__()
        self.input_format = input_format
        random.seed(42)
        # standard augmentations
        if training:
            self.transform_list = T.AugmentationList([
                T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1296,
                                     sample_style='choice'),
                T.RandomFlip()
            ])
        else:
            self.transform_list = T.AugmentationList([
                T.ResizeShortestEdge(short_edge_length=(966, 966), max_size=1296, sample_style='choice')])

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # read rgb image from disk
        image = utils.read_image(dataset_dict["file_name"], format=self.input_format)

        # rgb augment
        aug2_input = T.AugInput(image)
        transform2 = self.transform_list(aug2_input)
        image_aug = aug2_input.image

        # convert to tensor
        dataset_dict["image"] = torch.from_numpy(image_aug.transpose(2, 0, 1).copy())

        if "annotations" in dataset_dict:
            # augment labels according to image augmentation
            annos = [
                utils.transform_instance_annotations(obj, transform2, dataset_dict["image"].shape[1:])
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            # convert annotations to detectron2 format and filter empty ones
            instances = utils.annotations_to_instances(annos, dataset_dict["image"].shape[1:])
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
