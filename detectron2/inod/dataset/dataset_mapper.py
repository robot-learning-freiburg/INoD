"""
Detectron2 specific dataset definition for INoD pre-training.
author: Julia Hindel
adapted from https://detectron2.readthedocs.io/en/latest/_modules/detectron2/data/datasets/pascal_voc.html
"""

import os
import copy
from .dataset import ConcatDataset, get_filelist


def prep_source(dirname, file):
    """
       Create dummy Pascal VOC detection annotations for Detectron2 model.
       Real annotations will be computed on the fly during training.
       This set-up is required for standard detectron2 training.

       Args:
           dirname: Contains images.
           file (str): one of "train.txt" / "val.txt" which provides standard ImageNet dataset format.
       """

    # get all files from global dataset
    files = get_filelist(dirname, file)

    print(dirname, file, len(files))
    dicts = []
    id = 0

    for file in files:
        rgb_file = os.path.join(dirname, 'train', file)

        # save no height and width as not required here and done in data loading
        r = {"file_name": rgb_file, "height": 224, "width": 224, "image_id": id}
        dicts.append(r)
        id += 1

    print("FINISHED PREP_SOURCE")
    return dicts


class DatasetMapperINoD(ConcatDataset):
    """ Detectron2 mapper for dataset """

    def __init__(self, cfg):
        # init common dataset
        # in_pretrained = True if (("IN" in cfg.CONFIG_RES) or ("IN" in cfg.DATA_SOURCE)) else False
        RGB = False if "MSRA" in cfg.CONFIG_RES else True

        super().__init__(source_path=cfg.DATA_SOURCE, noise_path=cfg.DATA_NOISE, crop_size=cfg.INPUT.MIN_SIZE_TRAIN[0],
                         random_noise=cfg.RANDOM_NOISE, RGB=RGB, noise_split=cfg.NOISE_SPLIT)
        self.img_size = cfg.INPUT.MIN_SIZE_TRAIN[0]
        # don't need it as using prep_source method above for detectron2 to iterate
        del self.source_filelist, self.source_path
        print("INIT MAPPER")

    def __call__(self, dataset_dict):
        """ prepare training sample"""
        dataset_dict = copy.deepcopy(dataset_dict)
        # augment source and ndvi mask in the same way
        dataset_dict["source_image"] = self.transform_source(self.read_image(dataset_dict["file_name"]))
        dataset_dict["noise_image"] = self.transform_noise(self.get_random_noise())

        # set height and width of image
        dataset_dict["height"] = self.img_size
        dataset_dict["width"] = self.img_size

        return dataset_dict
