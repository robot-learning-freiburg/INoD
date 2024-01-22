"""
INoD ResNet-50 Encoder
author: Julia Hindel, largely adopted from detectron2: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved. (https://github.com/facebookresearch/detectron2)
"""

import torch
import math
import copy
import random

# detectron2
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, FPN
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

# local imports
from .backbone_utils import insert_noise


class INoDBackbone(Backbone):
    """
    Build a INoD pre-training backbone
    """

    def __init__(self, crop_size, device, slow_moving=False, noise_rate=0.2,
                 clean_image_ratio=0.5, noise_std=0.4, noisy_layers=['res2', 'res3', 'res4', 'res5'],
                 ref=-1, file="configs/backbone/bn_backboneC5.yaml", box_fill_rate=1 / 3):
        """

        :param crop_size: size of image crop
        :param input_shape: input shape for ResNet-50 network
        :param device: device to run model on (GPU/CPU)
        :param slow_moving: boolean; create second identical model for INoD pre-training precise batch norm calculation.
        :param noise_rate: relative percentage of noise that is injected.
        :param clean_image_ratio: no. of sample in a batch that should be replaced by zero-masks.
        :param noise_std: noise standard deviation; used to ensure noise only deviates by a certain number from the threshold
        :param noisy_layers: determines convolutional blocks after which noise is injected.
        :param ref: reference noise mask layer
        :param file: config file of ResNet-50 backbone
        """
        super(INoDBackbone, self).__init__()

        # set parameters
        self.slow_moving = slow_moving
        self.device = device
        self.clean_image_ratio = clean_image_ratio
        self.noise_std = noise_std
        self.noisy_layers = noisy_layers

        self.noise_t = torch.Tensor([noise_rate]).to(self.device)
        self.mask_shape = {}
        self.max_iter_rec_masks = 10
        self.box_fill_rate = box_fill_rate

        # define input shape for ResNet-50 stem if not defined
        input_shape = ShapeSpec(channels=3, height=crop_size, width=crop_size)

        self.cfg = get_cfg()
        self.cfg.merge_from_file(file)
        self.base_encoder = build_resnet_backbone(self.cfg, input_shape)
        print("R50 backbone")

        # copy output parameters of ResNet-50 backbone so FPN can be added
        self._out_features = self.base_encoder._out_features
        self._out_feature_channels = self.base_encoder._out_feature_channels
        self._out_feature_strides = self.base_encoder._out_feature_strides

        # initialize size of noisy layers with output size of intermediate ResNet-50 layers
        for i in self.noisy_layers:
            # create mask of BxHxW
            axis_length = math.ceil(crop_size / self.base_encoder._out_feature_strides[i])
            self.mask_shape[i] = (axis_length, axis_length)

        # select ref size
        if ref >= 0:
            self.ref_mask_shape = (7, 7)
        else:
            ref_mask = self.noisy_layers[ref]
            self.ref_mask_shape = self.mask_shape[ref_mask]

        # set min and max scale and noise
        self.max_scale = (self.ref_mask_shape[0] // 3)
        self.min_scale = 2 / 3
        self.max_noise = self.ref_mask_shape[0] * self.ref_mask_shape[1] * self.noise_t

        # init weights with ImageNet
        if self.cfg.MODEL.WEIGHTS:
            DetectionCheckpointer(self.base_encoder).load(self.cfg.MODEL.WEIGHTS)

    def forward(self, list):
        """ one forward pass of the ResNet-50 encoder """
        outputs = {}

        # disentangle list
        source = list[0]
        noise = list[1]
        mask = list[2]

        # compute noise features
        with torch.no_grad():  # no gradient to noise
            if self.slow_moving:
                base_encoder_m = copy.deepcopy(self.base_encoder)
                noise = base_encoder_m(noise)
                del base_encoder_m
            else:
                self.base_encoder.eval()
                noise = self.base_encoder(noise)
                self.base_encoder.train()

        # step-wise traversal of source through backbone
        for name, stage in self.base_encoder.named_children():
            source = stage(source)
            # insert noise if noise layer
            if name in self.noisy_layers:
                noise_i = noise[name]
                source = insert_noise(mask[name], source, noise_i)
            outputs[name] = source

        return outputs

    @torch.no_grad()
    def _momentum_update_base_encoder(self):
        """
        Update of the noise encoder, copy all weights
        """
        for param_q, param_k in zip(self.base_encoder.parameters(), self.base_encoder_m.parameters()):
            param_k.data = param_q.data  # param_k.data * self.m + param_q.data * (1. - self.m)

    def retrieve_noise_mask(self, batch_size):
        """ compute noise mask """

        no_boxes = random.randint(3 * batch_size, 5 * batch_size)
        boxes = torch.rand(no_boxes, 3, 3)
        bin_boxes = (boxes <= self.box_fill_rate).float()

        scale_x = (self.max_scale - self.min_scale) * torch.rand(no_boxes) + self.min_scale
        scale_y = scale_x * ((4 - 0.25) * torch.rand(no_boxes) + 0.25)
        position = (self.ref_mask_shape[0] - 2) * torch.rand(no_boxes, 2)
        reference_noise_mask = torch.zeros(batch_size, *self.ref_mask_shape)
        idx = 0
        for i in range(no_boxes):
            box = \
            torch.nn.functional.interpolate(bin_boxes[i].unsqueeze(0).unsqueeze(0).float(), recompute_scale_factor=True,
                                            scale_factor=(scale_x[i].item(), scale_y[i].item()), mode='area')[0, 0]
            noise_rate = len(torch.nonzero(box))
            if (noise_rate == 0) or (noise_rate > self.max_noise):
                continue
            box = (box >= 0.5).float()
            limit_x, limit_y = box.shape[0], box.shape[1]
            pos_end_x = int(position[i][0]) + limit_x
            pos_end_y = int(position[i][1]) + limit_y
            if pos_end_x >= self.ref_mask_shape[0]:
                limit_x = self.ref_mask_shape[0] - 1 - int(position[i][0])
                pos_end_x = self.ref_mask_shape[0] - 1
            if pos_end_y >= self.ref_mask_shape[1]:
                limit_y = self.ref_mask_shape[1] - 1 - int(position[i][1])
                pos_end_y = self.ref_mask_shape[1] - 1
            if limit_x > 0 and limit_y > 0:
                box = box[0:limit_x, 0:limit_y]
                if len(torch.nonzero(box)) == 0:
                    continue
                if idx < 2 * (batch_size - 1):
                    if idx == 0:
                        frame = idx
                    else:
                        frame = idx % (batch_size - 1)
                else:
                    frame = random.randint(0, batch_size - 1)
                count = 0
                while len(torch.nonzero(box)) + len(torch.nonzero(reference_noise_mask[frame])) > self.max_noise:
                    if batch_size == 1:
                        count = self.max_iter_rec_masks
                        break
                    frame = (frame + 1) % (batch_size - 1)
                    count += 1
                    if count == self.max_iter_rec_masks:
                        break
                if count < self.max_iter_rec_masks:
                    idx += 1
                    reference_noise_mask[frame, int(position[i][0]):int(pos_end_x),
                    int(position[i][1]):int(pos_end_y)] = box

        reference_noise_mask = reference_noise_mask.to(self.device)

        # get clean_image_ratio x batch_size images and set their mask to false -> no noise injection
        original = torch.randperm(batch_size)[:(int(batch_size * self.clean_image_ratio))]
        if batch_size == 1:
            original = [0] if torch.rand(1) < self.clean_image_ratio else []
        reference_noise_mask[original] = 0

        # create random mask according to which final mask will be split (splitting mask)
        split = torch.rand(batch_size, *self.ref_mask_shape, device=self.device)
        lower_portion = 0
        diff = 1 / len(self.noisy_layers)
        upper_portion = diff

        # define noise masks
        noise_masks = {}

        for name, shape in self.mask_shape.items():
            # compute random mask of size BxHxW and threshold it in x interludes
            split_mask = torch.where((lower_portion <= split) & (split < upper_portion), torch.ones_like(split),
                                     torch.zeros_like(split))
            # mutliply splitting mask per layer with reference noise mask
            temp_mask = split_mask * reference_noise_mask

            # interpolate noise mask to correct size of layer
            noise_masks[name] = torch.nn.functional.interpolate(temp_mask.unsqueeze(0).float(),
                                                                size=(shape[0], shape[1]),
                                                                mode='nearest')[0]
            # need to interpolate in the other way to make it consistent in reference noise mask
            if shape[0] < self.ref_mask_shape[0]:
                reference_noise_mask += torch.nn.functional.interpolate(noise_masks[name].unsqueeze(0),
                                                                        size=(
                                                                            self.ref_mask_shape[0],
                                                                            self.ref_mask_shape[0]),
                                                                        mode='nearest')[0]
            lower_portion = upper_portion
            upper_portion += diff

        # make binary
        reference_noise_mask = torch.clamp(reference_noise_mask, min=0, max=1)

        return noise_masks, reference_noise_mask


@BACKBONE_REGISTRY.register()
def build_INoD_fpn_backbone(cfg, input_shape):
    """
    FPN including INoD encoder based on ResNet-50
    Args:
        cfg: a detectron2 cfg file
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    # build bottom-up according to INoD
    bottom_up = INoDBackbone(cfg.INPUT.MIN_SIZE_TRAIN[0], cfg.MODEL.DEVICE, clean_image_ratio=0,
                         noisy_layers=cfg.NOISY_LAYERS, ref=cfg.NOISY_REF,
                         file=cfg.CONFIG_RES, slow_moving=True, noise_rate=cfg.NOISE_RATE, box_fill_rate=cfg.BOX_FILL_RATE)
    # normal FPN
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
