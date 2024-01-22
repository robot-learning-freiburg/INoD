"""
INoD Encoder Utils
author: Julia Hindel
"""

import torch
import cv2
import numpy as np

# detectron2
from detectron2.structures import BoxMode
from detectron2.data.detection_utils import annotations_to_instances


def insert_noise(mask, source, noise):
    """ Feature Combination """
    # revert order, so channel is first (BxCxHxW -> CxBxHxW)
    noise = noise.permute(1, 0, 2, 3).contiguous()
    source = source.permute(1, 0, 2, 3).contiguous()
    # convert mask to boolean
    mask = mask > 0

    # multiply every channel of noise with mask (CxBxHxW * BxHxW) (mask: True: noise, False: 0)
    noise = noise * mask
    # multiply every channel of source with inverted mask (CxBxHxW * BxHxW) (mask: True: 0, False: source)
    inverted_mask = ~mask
    source = source * inverted_mask

    # add source and noise to produce final feature_map
    feature_map = source + noise
    # revert order, so batch_size is first (CxBxHxW -> BxCxHxW)
    feature_map = feature_map.permute(1, 0, 2, 3).contiguous()

    return feature_map

def retrieve_bb_interpolate(mask_final, image_size, device):
    shape = image_size[0]
    mask_interpolated = torch.nn.functional.interpolate(mask_final.unsqueeze(0).float(), size=(shape[0], shape[1]),
                                                        mode='area').cpu()
    bb = []
    for i in mask_interpolated[0]:
        single_mask = i.detach().numpy()
        # avoid them being stuck
        single_mask = cv2.erode(single_mask, np.ones((2, 1), np.uint8), iterations=1)
        single_mask = cv2.convertScaleAbs(single_mask)
        contours, hierarchy = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bb_maski = []
        if contours:
            for c in contours:
                if len(c) >= 3:
                    x, y, w, h = cv2.boundingRect(c)
                    c = [c.reshape(-1).tolist()]
                    bb_maski.append([[x, y, x + w, y + h], c])
                else:
                    print('opencv error', c)
        bb.append(bb_maski)

    # format: no. in batch, y1, x1, no. in batch, y2, x2
    class_names = ["noise"]
    gt_instances = []
    for i in range(len(image_size)):
        if not bb[i]:
            gt_instances.append(annotations_to_instances([], image_size[i]).to(device))
            continue
        bb_i = bb[i] # torch.stack(bb[i])
        instances = []
        for obj in bb_i:
            cls = "noise"
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox = [obj[0][0], obj[0][1], obj[0][2], obj[0][3]]
            segmentation = obj[1]
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "segmentation": segmentation, "bbox_mode": BoxMode.XYXY_ABS}
            )
        gt_instances.append(annotations_to_instances(instances, image_size[i]).to(device))

    return gt_instances
