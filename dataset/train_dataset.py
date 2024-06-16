import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch


class TrainDataset(data.Dataset):
    def __init__(self, image_root, gt_root, image_size):
        self.image_size = image_size

        print("Read Files in Directory")
        self.images = [os.path.join(image_root, f) for f in os.listdir(
            image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(
            gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        print("Sort Files")
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
