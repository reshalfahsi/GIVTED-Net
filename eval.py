import torch
import torch.nn.functional as F

import time
import numpy as np

from tqdm import tqdm
from ptflops import get_model_complexity_info

import argparse
import os

import cv2

from tools.eval import available_dataset, evaluate
from dataset.test_dataset import TestDataset
from givtednet.model import GIVTEDNet


def parse_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Testing configuration.")

    # Add arguments
    parser.add_argument("--image_size", type=int, default=224, help="Testing image size.")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Small value for numerical stability.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for prediction.")

    # Parse arguments from the command line
    return parser.parse_args()


def eval_fn():
    # Parse arguments
    config = parse_arguments()

    print_once_param = True
    warmup_counter = 20

    for _data_name in available_dataset:

        pth_path = f"./experiment/{_data_name}/model_pth/GIVTEDNet_best.pth"

        model = GIVTEDNet()
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(pth_path))
            model.cuda()
        else:
            model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))

        if print_once_param:
            macs, params = get_model_complexity_info(
                model,
                (3, config.image_size, config.image_size),
                as_strings=True,
                print_per_layer_stat=True,
                verbose=True,
            )
            print("\n\n=======================================================")
            print("{:<30}  {:<8}".format("Computational complexity: ", macs))
            print("{:<30}  {:<8}".format("Number of parameters: ", params))
            print("=======================================================\n\n")

            print_once_param = False

        if torch.cuda.is_available():
            x = torch.rand(1, 3, config.image_size, config.image_size).cuda()
        else:
            x = torch.rand(1, 3, config.image_size, config.image_size).cpu()

        model = torch.jit.trace(model, x)
        model.eval()

        with torch.no_grad():
            for _ in tqdm(range(warmup_counter)):
                if torch.cuda.is_available():
                    x = torch.rand(1, 3, config.image_size, config.image_size).cuda()
                else:
                    x = torch.rand(1, 3, config.image_size, config.image_size).cpu()
                y = model(x)
        

            data_path_ = f"./experiment/{config.dataset_name}/TestDataset"
            save_path_ = f"./experiment/{config.dataset_name}/result"

            os.makedirs(save_path_, exist_ok=True)

            image_root = "{data_path_}/images/"
            gt_root = "{data_path_}/masks/"

            if not os.path.exists(image_root) or not os.path.exists(gt_root):
                continue

            N = len(os.listdir(gt_root))
            test_loader = TestDataset(image_root, gt_root, config.image_size)

            DSC, IoU, Recall, Precision = list(), list(), list(), list()
            FPS = list()

            for i in tqdm(range(N)):
                image, gt, name = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= gt.max() + config.epsilon
                if torch.cuda.is_available():
                    image = image.cuda()
                else:
                    image = image.cpu()

                start = time.time()
                res = model(image)
                end = time.time()
                FPS.append(1. / (end - start))
                res = F.interpolate(res, size=gt.shape, mode="bilinear", align_corners=False)

                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = abs(res - res.min()) / (abs(res.max() - res.min()) + config.epsilon)

                dice, iou, recall, precision = evaluate(res, gt, config.threshold, config.epsilon)
                DSC.append(dice)
                IoU.append(iou)
                Recall.append(recall)
                Precision.append(precision)

                res[res >= config.threshold] = 1
                res[res < config.threshold] = 0

                cv2.imwrite(os.path.join(save_path_, name), res * 255)

            DSC = np.array(DSC)
            IoU = np.array(IoU)
            Recall = np.array(Recall)
            Precision = np.array(Precision)

            print(_data_name, "Finish!")
            print(f"Mean DICE: {DSC.mean():.4f}")
            print(f"Mean IoU: {IoU.mean():.4f}")
            print(f"Recall: {Recall.mean():.4f}")
            print(f"Precision: {Precision.mean():.4f}")
            print("----------------------------------------------------")

            image_root = ""
            gt_root = ""

    FPS = np.array(FPS)
    print(f"FPS: {FPS.mean():.4f}")
    print("----------------------------------------------------")
    print()
    print("Evaluation Complete")



if __name__ == "__main__":
    eval_fn()
