"""
Load Pascal VOC detection annotations to Detectron2 format.
author: Julia Hindel
largely adapted from https://detectron2.readthedocs.io/en/latest/_modules/detectron2/data/datasets/pascal_voc.html
"""

from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

import numpy as np
import os
import ast
import xml.etree.ElementTree as ET

def prep_SB16(dirname, dataset_split, split, mask_on, thing_classes_prep):
    """
       Load Pascal VOC detection annotations to Detectron2 format for Sugar Beets 2016.

       Args:
           dirname: Contain "ImageSets", annotation and ground truth files in base directory
           anno_type: "Mask" for collecting mask annotations from "pose"
           split (str): one of "train", "test", "val"
           type: indicating bb or masks
       """

    # read train file (list of xml files)

    with PathManager.open(os.path.join(dirname, dataset_split, split + ".txt")) as f:
        xml_files = np.loadtxt(f, dtype=str)

    print(dirname, split, len(xml_files))

    stats = {"weed": 0, "crop": 0, "weed_cnt": 0, "crop_cnt": 0}

    org_split = "train"

    dicts = []

    id = 0

    # iterate over xml files and
    for file in xml_files:
        file_name = os.path.join(dirname, org_split, file)
        anno_file = os.path.join(dirname, org_split, file[:-4] + ".xml")
        id += 1

        if not os.path.isfile(anno_file):
            continue

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": file_name,
            "image_id": id,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }

        instances = []

        # read all objects
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            dic_object = {"category_id": thing_classes_prep.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            stats[cls] += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            stats[f"{cls}_cnt"] += 1
            if mask_on:
                segmentation = ast.literal_eval(' '.join(obj.find("pose").text.split()).replace("  ", ","))

                if len(segmentation) == 0:
                    print('masks not provided')
                dic_object["segmentation"] = segmentation
            instances.append(dic_object)
        r["annotations"] = instances
        dicts.append(r)
    for c in thing_classes_prep:
        stats[c] = np.sqrt(stats[c] / stats[f"{c}_cnt"])
    print(split, stats)
    return dicts

def prep_FP22(dirname, dataset_split, split, mask_on, thing_classes_prep):
    """
       Load Pascal VOC detection annotations to Detectron2 format for CP 2022.

       Args:
           dirname: Contain "ImageSets", annotation and ground truth files in base directory.
           anno_type: "Mask" for collecting mask annotations from "pose".
           split (str): one of "train", "test", "val".
           type: indicating bb or masks.
       """

    # read train file (list of xml files)

    with PathManager.open(os.path.join(dirname, dataset_split, split + ".txt")) as f:
        xml_files = np.loadtxt(f, dtype=str)

    print(dirname, split, len(xml_files))

    org_split = "train"

    dicts = []

    id = 0
    stats = {"weed": 0, "potato": 0, "weed_cnt": 0, "potato_cnt": 0}

    # iterate over xml files and
    for file in xml_files:
        anno_file = os.path.join(dirname, org_split, file)
        file_name = os.path.join(dirname, org_split, file[:-4] + ".png")
        id += 1

        if not os.path.isfile(anno_file):
            continue

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": file_name,
            "image_id": id,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }

        instances = []

        # read all objects
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if "abd" in cls:
                print(file_name)
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            dic_object = {"category_id": thing_classes_prep.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            stats[cls] += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            stats[f"{cls}_cnt"] += 1
            if mask_on:
                segmentation = ast.literal_eval(' '.join(obj.find("pose").text.split()).replace("  ", ","))

                if len(segmentation) == 0:
                    print('masks not provided')
                dic_object["segmentation"] = segmentation
            instances.append(dic_object)
        r["annotations"] = instances
        dicts.append(r)
    for c in thing_classes_prep:
        stats[c] = np.sqrt(stats[c] / stats[f"{c}_cnt"])
    print(split, stats)
    return dicts
