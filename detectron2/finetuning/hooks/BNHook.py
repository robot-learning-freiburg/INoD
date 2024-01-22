"""
Precise BN computed before fine-tuning.
author: Julia Hindel, largely adopted from detectron2: Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved. (https://detectron2.readthedocs.io/en/latest/_modules/fvcore/nn/precise_bn.html)
"""

import logging
import itertools

# detectron2
from detectron2.engine.hooks import HookBase
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats


class BNHook(HookBase):
    def __init__(self, model, data_loader, num_iter):
        self._logger = logging.getLogger(__name__)
        if len(get_bn_modules(model)) == 0:
            self._logger.info(
                "PreciseBN is disabled because model does not contain BN layers in training mode."
            )
            self._disabled = True
            return

        self._model = model
        self._data_loader = data_loader
        self._data_iter = None
        self._num_iter = num_iter

    def before_train(self):

        if self._data_iter is None:
            self._data_iter = iter(self._data_loader)

        def data_loader():
            for num_iter in itertools.count(1):
                if num_iter % 100 == 0:
                    self._logger.info(
                        "Running precise-BN ... {}/{} iterations.".format(num_iter, self._num_iter)
                    )
                    print("Running precise-BN ... {}/{} iterations.".format(num_iter, self._num_iter))
                # This way we can reuse the same iterator
                yield next(self._data_iter)

        # with EventStorage():  # capture events in a new storage to discard them
        self._logger.info(
            "Running precise-BN for {} iterations...  ".format(self._num_iter)
            + "Note that this could produce different statistics every time."
        )
        print("Running precise-BN for {} iterations...  ".format(self._num_iter))

        update_bn_stats(self._model, data_loader(), self._num_iter)

        del self._data_iter, self._data_loader
