"""
Detectron2 visualizations on wandb.
author: Julia Hindel

"""
import wandb
import cv2


def visualize_test_data_wandb(data_loader, predictor, cfg, MetadataCatalog, limit=10):
    """
    Prepare visualization for wandb
    :param data_loader: detectron2 dataloader
    :param predictor: trained model
    :param cfg: detectron2 config
    :param MetadataCatalog: detectron2 metadatacatalog
    :param limit: no. of images to display
    :return:
    """
    # init storage
    i = 0
    img_array = []

    # retrieve class names
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    class_id_to_label = {int(v): k for v, k in enumerate(class_names)}

    # iterate data loader, get predicitions and push to wandb
    for d in data_loader:
        dict = d[0]
        img = dict["image"].permute(1, 2, 0).cpu().detach().numpy()
        outputs = predictor(img)
        # print(outputs)
        if outputs is None:
            print("NO PREDICTIONS")
        img, all_boxes, all_pred_boxes = prep_img_wandb(dict, img, outputs)
        img_array.append(wandb.Image(img, caption=dict["image_id"], boxes={"ground_truth": {"box_data": all_boxes,
                                                                                            "class_labels": class_id_to_label},
                                                                           "predictions": {"box_data": all_pred_boxes,
                                                                                           "class_labels": class_id_to_label},
                                                                           }))
        i += 1
        if i > limit:
            break

    wandb.log({"Example testing images": img_array})


def prep_img_wandb(dict, img, outputs=None):
    """
    Prepare annotations for wandb format
    :param dict: instance format of detectron2
    :param img: orginial image (required for size)
    :param outputs: predictions of model
    :return:
    """
    img = img[:, :, [2, 1, 0]]
    # add padding around images so it's easier to display
    bottom = (1400 - img.shape[0])
    right = (1400 - img.shape[1])
    img = cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    class_id = dict["instances"].gt_classes.detach().numpy()
    boxes = dict["instances"].gt_boxes.tensor.detach().numpy()

    all_boxes = []
    # gt boxes
    for b_i, box in enumerate(boxes):
        # get coordinates and labels with new image shape
        box_data = {"position": {
            "minX": box[0] / img.shape[1],
            "maxX": box[2] / img.shape[1],
            "minY": box[1] / img.shape[0],
            "maxY": box[3] / img.shape[0]},
            "class_id": int(class_id[b_i])}
        all_boxes.append(box_data)

    if outputs:
        # obtain predictions
        pred_class_id = outputs["instances"].pred_classes.to("cpu")
        pred_boxes = outputs["instances"].pred_boxes.to("cpu")
        scores = outputs["instances"].scores.to("cpu")

        all_pred_boxes = []

        # convert predictions to new image shape
        for b_i, box in enumerate(pred_boxes):
            # get coordinates and labels
            box_data = {"position": {
                "minX": box[0].numpy() / img.shape[1],
                "maxX": box[2].numpy() / img.shape[1],
                "minY": box[1].numpy() / img.shape[0],
                "maxY": box[3].numpy() / img.shape[0]},
                "class_id": int(pred_class_id[b_i].numpy()),
                "scores": {"score": scores[b_i].item()}}
            all_pred_boxes.append(box_data)

        return img, all_boxes, all_pred_boxes

    return img, all_boxes
