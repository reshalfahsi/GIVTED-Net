import torch.utils.data as data

from dataset.train_dataset import TrainDataset


def get_loader(
        dataset,
        batchsize,
        shuffle=True,
        num_workers=2,
        pin_memory=True):
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return data_loader


def get_dataset(image_root, gt_root, image_size):
    return TrainDataset(image_root, gt_root, image_size)
