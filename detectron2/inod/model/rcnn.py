"""
INoD RCNN Model
author: Julia Hindel, largely adopted from detectron2: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved. (https://github.com/facebookresearch/detectron2)
"""

import numpy as np
from typing import Dict, List, Optional
import torch
from torch import nn
import gc

# detectron2
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

# local imports
from inod.model.backbone_utils import insert_noise, retrieve_bb_interpolate


@META_ARCH_REGISTRY.register()
class INoDRCNN(nn.Module):
    """
    Generalized R-CNN used for INoD Pre-training.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
            input_format: Optional[str] = None,
            vis_period: int = 0,
            device,
            distributed: bool = True
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
            device: device for visualizations
            distributed: distributed training
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        self.device = device
        self.dump_patches = True
        self.distributed = distributed
        self.iter = 0
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "device": cfg.MODEL.DEVICE,
            "distributed": cfg.DISTRIBUTED
        }

    def _move_to_current_device(self, x):
        # JH: adapted, short cut to move device
        return x.to(self.device)

    def visualize_training(self, batched_inputs, gt_instances, mask_final, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 40 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 40

        for input, prop, mask, instances in zip(batched_inputs, proposals, mask_final, gt_instances):
            img_source = input["source_image"].unsqueeze(0)
            img_noise = input["noise_image"].unsqueeze(0)

            mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                                   size=(img_source.shape[2], img_source.shape[3]),
                                                   mode='nearest')[0]
            if not self.distributed:
                mask = mask.cpu()
            img = insert_noise(mask, img_source, img_noise)
            img = img[0].cpu().permute(1, 2, 0).numpy()[:, :, 0]
            del img_source, img_noise, mask
            # img = img[0].cpu().permute(1, 2, 0).numpy()
            v_gt = Visualizer(img, None)
            boxes = instances.gt_boxes.tensor.to("cpu").numpy()
            if instances.gt_classes.shape[0] > 0:
                masks = instances.gt_masks
                v_gt = v_gt.overlay_instances(boxes=boxes, masks=masks)
            else:
                v_gt = v_gt.overlay_instances(boxes=boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        # print("FORWARD PASS")
        # print("START ITER", torch.cuda.memory_allocated(0))

        if not self.training:
            return self.inference(batched_inputs)

        # preprocess all components
        source_images, noise_images = self.preprocess_image(batched_inputs)

        mask, mask_final = self.backbone.bottom_up.retrieve_noise_mask(len(source_images.image_sizes))

        # retrieve bounding boxes (changed for masks)
        gt_instances = retrieve_bb_interpolate(mask_final, source_images.image_sizes, self.device)

        # get feature maps after encoder (including FPN)
        features = self.backbone((source_images.tensor, noise_images.tensor, mask, mask_final))

        # remove to save space
        del noise_images, mask

        # remaining Faster R-CNN components (RPN, Detector Head)
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(source_images, features, gt_instances)


        _, detector_losses = self.roi_heads(source_images, features, proposals, gt_instances)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, gt_instances, mask_final, proposals)

        # compute loss
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        del _, mask_final, source_images, features, proposals, detector_losses, proposal_losses, batched_inputs
        torch.cuda.empty_cache()
        gc.collect()
        return losses

    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
    ):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
        Returns:
            A list[Instances] containing network outputs.
        """
        assert not self.training

        # preprocess images
        source_images, noise_images = self.preprocess_image(batched_inputs)
        # compute masks
        mask, mask_final = self.backbone.bottom_up.retrieve_noise_mask(len(source_images.image_sizes))

        # retrieve bounding box
        gt_instances = retrieve_bb_interpolate(mask_final, source_images.image_sizes, self.device)
        batched_inputs[0]["reference_noise_mask"] = mask_final

        # get feature maps after encoder (including FPN)
        with torch.no_grad():
            features = self.backbone((source_images.tensor, noise_images.tensor, mask))

            # remove to save space
            del noise_images, mask

            # remaining Faster R-CNN components (RPN, Detector Head)
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(source_images, features, None)
            results, _ = self.roi_heads(source_images, features, proposals, None)

            size = source_images.image_sizes
            del source_images, _, proposals, features

        return INoDRCNN._postprocess(results, batched_inputs, gt_instances, size)

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Move all images to training device and convert to tensors.
        """
        source_images = [self._move_to_current_device(x["source_image"]) for x in batched_inputs]
        source_images = ImageList.from_tensors(source_images, self.backbone.size_divisibility)

        noise_images = [self._move_to_current_device(x["noise_image"]) for x in batched_inputs]
        noise_images = ImageList.from_tensors(noise_images, self.backbone.size_divisibility)

        return source_images, noise_images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], gt_instances, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, gt_instance, image_size in zip(
                instances, batched_inputs, gt_instances, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r, "gt": gt_instance})
        return processed_results
