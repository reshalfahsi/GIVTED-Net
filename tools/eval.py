import numpy as np


available_dataset = [
    "ISIC2018",
    "KvasirInstrument",
    "WBCImage",
]


def evaluate(pred, gt, threshold=0.5, epsilon=1e-8):
    """Evaluate the prediction performance.

    :param pred: predicted mask
    :param gt: ground-truth mask
    :return: Evaluation metrics
    """

    label_pred = np.zeros_like(pred)

    label_pred[pred > threshold] = 1

    obj_pred = np.sum(label_pred == 1)

    label_gt = np.zeros_like(gt)

    label_gt[gt > threshold] = 1

    obj_gt = np.sum(label_gt == 1)

    TP = np.sum((label_pred == 1) & (label_gt == 1))
    FN = abs(obj_gt - TP)
    FP = abs(obj_pred - TP)

    iou = abs(TP) / (abs(TP + FN + FP) + epsilon)
    dice = abs(2 * TP) / (abs(2 * TP + FP + FN) + epsilon)

    TN = np.sum((label_pred == 0) & (label_gt == 0))
    recall = TP/(TP + FN)
    precision = TP/(TP + FP)

    return dice, iou, recall, precision
