"""
Convert INoD Pre-trained model to detectron2 fine-tuning.
author: Julia Hindel, largely adopted from MoCo: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved. (https://github.com/facebookresearch/moco)
"""

import pickle as pkl
import sys
import torch

if __name__ == "__main__":
    # first param: model name before
    input = sys.argv[1]

    # load model
    obj = torch.load(input, map_location="cpu")
    obj = obj["model"]

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("backbone."):
            # remove other Faster R-CNN parts
            # if ("roi_heads" in k) or ("rpn_head" in k):
            #    continue
            print(k, "->", k)
            newmodel[k] = v.numpy()
            continue
        # delete second base encoder (required for precise batch norm)
        if "base_encoder_m" in k:
            continue
        # copy weights and re-name
        old_k = k
        k = k.replace("base_encoder.", "")
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {"model": newmodel, "__author__": "JH", "matching_heuristics": True}

    # second param: model name after conversion
    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
