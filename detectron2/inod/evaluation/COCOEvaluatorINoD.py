"""
Evaluation of INoD Pre-training.
author: Julia Hindel, largely adopted from detectron2: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved. (https://github.com/facebookresearch/detectron2)
"""

import copy
import itertools
import os
import random
import numpy as np
from collections import OrderedDict
import torch
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util


import detectron2.utils.comm as comm
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco
try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval


class COCOEvaluatorINoD(COCOEvaluator):
    """
    COCO Evaluator for INoD Detection. Overwrites Required Methods from original detectron2 COCOEvaluator
    """

    def __init__(
            self,
            dataset_name,
            tasks=None,
            distributed=True,
            output_dir=None,
            *,
            max_dets_per_image=None,
            use_fast_impl=True,
            kpt_oks_sigmas=(),
            allow_cached_coco=True,
    ):
        super().__init__(dataset_name=dataset_name, tasks=tasks,
                         distributed=distributed,
                         output_dir=output_dir,
                         max_dets_per_image=max_dets_per_image,
                         use_fast_impl=use_fast_impl,
                         allow_cached_coco=allow_cached_coco)
        # added JH: always to evaluation
        self._do_evaluation = True
        # added JH: no annotations needed
        self._coco_api.dataset['annotations'] = []
        self._rand = comm.get_rank() * 1000000
        if self._do_evaluation:
            self._kpt_oks_sigmas = kpt_oks_sigmas

    def reset(self):
        # remove all ground truth and predictions every time
        self._predictions = []
        # added JH: no annotations provided at the start
        self._coco_api.dataset['annotations'] = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: batched_inputs (detectron2 instance format), including ground truth bounding boxes
             which have been computed on the fly.
            outputs: the predictions of the INoD detection model.
        """

        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                pred = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(pred, input["image_id"], self._rand)
            # overwrite ground truth with ground truth bounding boxes from masks
            if "gt" in output:
                gt = output["gt"].to(self._cpu_device)
                self._coco_api.dataset['annotations'] += instances_to_coco_json(gt, input["image_id"], self._rand,
                                                                                coco_api=self._coco_api)
            else:
                print("ERROR, instances not received")
            # save proposals (if there)
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            gt = comm.gather(self._coco_api.dataset['annotations'], dst=0)
            self._coco_api.dataset['annotations'] = list(itertools.chain(*gt))
            print("SUM", len(gt), len(self._coco_api.dataset['annotations']))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            # added JH, need to create new index every time as ground truth bounding boxes have changed
            self._coco_api.createIndex()
            self._eval_predictions(predictions, img_ids=img_ids)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)


def instances_to_coco_json(instances, img_id, rand, coco_api=None):
    """
    Convert an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []
    if coco_api:
        boxes = instances.gt_boxes.tensor.numpy()
        classes = instances.gt_classes.tolist()
        id = len(coco_api.dataset['annotations']) + rand

        has_mask = instances.has("gt_masks")
        if has_mask:
            # use RLE to encode the masks, because they are too large and takes memory
            # since this evaluator stores outputs of the entire dataset
            # rles = instances.gt_masks
            rles = [mask for mask in instances.gt_masks.polygons]
        else:
            rles = []

    else:
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.tolist()
        scores = instances.scores.tolist()
        has_mask = instances.has("pred_masks")
        if has_mask:
            # print("print masks", instances.pred_masks)
            # use RLE to encode the masks, because they are too large and takes memory
            # since this evaluator stores outputs of the entire dataset
            rles = [
                mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                for mask in instances.pred_masks
            ]
            for rle in rles:
                # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                # json writer which always produces strings cannot serialize a bytestream
                # unless you decode it. Thankfully, utf-8 works out (which is also what
                # the pycocotools/_mask.pyx does).
                rle["counts"] = rle["counts"].decode("utf-8")
        else:
            rles = []

    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()

    if coco_api:
        annotations = []
        for i, bb in enumerate(boxes):
            id += 1
            entry = {"id": id,
                 "image_id": img_id,
                 "bbox": bb,
                 "area": bb[2] * bb[3],
                 "iscrowd": 0,
                 'category_id': classes[i]}
            if has_mask:
                entry["segmentation"] = rles[i]
            annotations.append(entry)
        return annotations

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        results.append(result)
    return results
