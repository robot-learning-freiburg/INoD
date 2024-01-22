import torch
import wandb
import os
import numpy as np
import cv2
import re

from .model.backbone_utils import insert_noise
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer, GenericMask


class MyVisualizer(Visualizer):
    def _jitter(self, color):
        return color

    def overlay_instances(
            self,
            *,
            boxes=None,
            labels=None,
            masks=None,
            keypoints=None,
            assigned_colors=None,
            alpha=0.5,
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.
        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = 0
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                if alpha == 0.5:
                    self.draw_box(boxes[i], edge_color=color, alpha=1.0, line_style="--")
                else:
                    self.draw_box(boxes[i], edge_color=color, alpha=1.0, line_style="-")

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    # skip small mask without polygon
                    if len(masks[i].polygons) == 0:
                        continue

                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                        instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                        or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                        np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                        * 0.5
                        * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output


def visualize_test_data_INoD(data_loader, predictor, cfg, MetadataCatalog, name, mask=False, limit=200):
    i = 0
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN)
    metadata.thing_classes = ["noise"]
    colors = {"noise": (1, 0, 0)}
    colors_pred = {"noise": (0, 1, 0)}

    if mask:
        folder = "INoD_ISEG"
    else:
        folder = "INoD_BB"
    if not os.path.isdir(f"{name}/{folder}"):
        os.makedirs(f"{name}/{folder}/rgb")
        os.makedirs(f"{name}/{folder}/pred_gt")

    for d in data_loader:
        outputs = predictor(d[0])
        img = torch.nn.functional.interpolate(d[0]["reference_noise_mask"].unsqueeze(0), (224, 224))[0, 0]
        img = (img.cpu().numpy() * 255).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        filename = re.split('/', d[0]["file_name"])[-1]
        cv2.imwrite(f"{name}/{folder}/rgb/{filename}_{i}.png", img)
        v = MyVisualizer(img, metadata=metadata, scale=1,
                         instance_mode=ColorMode.SEGMENTATION)
        v._default_font_size = 10
        target_fields = outputs[0]["gt"].get_fields()
        labels = ['noise' for i in target_fields["gt_classes"]]
        assigned_color = [colors[i] for i in labels]
        boxes = outputs[0]["gt"].gt_boxes.tensor.to("cpu").numpy()
        if mask and hasattr(outputs[0]["gt"], 'gt_masks'):
            masks = outputs[0]["gt"].gt_masks
            v.overlay_instances(boxes=boxes, masks=masks, assigned_colors=assigned_color, alpha=0.5)
        else:
            v.overlay_instances(boxes=boxes, assigned_colors=assigned_color, alpha=0.5)
        pred_class_id = outputs[0]["instances"].pred_classes.to("cpu")
        pred_boxes = outputs[0]["instances"].pred_boxes.to("cpu").tensor.numpy()
        pred_labels = ['noise' for i in pred_class_id]
        assigned_color_pred = [colors_pred[i] for i in pred_labels]
        if mask and (len(outputs[0]["instances"].pred_masks) > 0):
            masks = outputs[0]["instances"].pred_masks.cpu().numpy()
            v.overlay_instances(boxes=pred_boxes, masks=masks, assigned_colors=assigned_color_pred, alpha=0)
        else:
            v.overlay_instances(boxes=pred_boxes, assigned_colors=assigned_color_pred, alpha=1.0)
        pred = cv2.cvtColor(v.output.get_image(), cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{name}/{folder}/pred_gt/{filename}_{i}.png", pred)
        i += 1
        if i > limit:
            break
