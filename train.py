import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import argparse
import os

from givtednet.model import GIVTEDNet
from tools.train import train_epoch
from tools.utils import EarlyStopping, AvgMeter
from tools.loss import LossFunction
from dataset.loader import get_dataset, get_loader


def parse_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Training configuration.")

    # Add arguments
    parser.add_argument("--epoch", type=int, default=100, help="Training epochs.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2, help="Learning rate for training.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the model.")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for SGD.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch sizes.")
    parser.add_argument("--image_size", type=int, default=224, help="Training image size.")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Small value for numerical stability.")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--lr_scheduler_cooldown", type=int, default=8, help="Learning rate scheduler cooldown.")
    parser.add_argument("--step_lr", type=int, default=3, help="How many step to reduce the learning rate once the performance degrades during training.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name.")

    # Parse arguments from the command line
    return parser.parse_args()


def train_fn():
    # Parse arguments
    config = parse_arguments()

    best = 0.0

    train_path = f"./experiment/{config.dataset_name}/TrainDataset"
    train_save = f"./experiment/{config.dataset_name}/model_pth"

    os.makedirs(train_save, exist_ok=True)

    model = GIVTEDNet(config.dropout)

    weight_pth = os.path.join(train_save, "GIVTEDNet.pth")
    if os.path.exists(weight_pth):
        print(f"Weight found: {weight_pth}")
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weight_pth))
        else:
            model.load_state_dict(torch.load(weight_pth, map_location=torch.device('cpu')))

    if torch.cuda.is_available():
        model.cuda()

    criterion = LossFunction(config.epsilon)

    optimizer = torch.optim.SGD(
        model.parameters(),
        config.lr,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
        nesterov=True,
    )

    image_root = f"{train_path}/images/"
    gt_root = f"{train_path}/masks/"

    dataset = get_dataset(
        image_root,
        gt_root,
        image_size=config.image_size,
    )
    n_train = int(len(dataset) * 0.9)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    loss_train = [0]
    loss_val = [0]

    train_loader = get_loader(train_set, config.batch_size)
    val_loader = get_loader(val_set, config.batch_size)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'max',
        patience=config.step_lr,
        cooldown=config.lr_scheduler_cooldown,
        verbose=True,
    )

    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for n in range(1, config.epoch + 1):
        loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer, 
            n, 
            config.epoch,
            config.batch_size, 
            total_step
        ).data.cpu().numpy()
        loss_train.append(loss)
        val_score = 0

        loss_record = AvgMeter()

        for i, pack in enumerate(val_loader, start=1):
            model.eval()

            images, gts = pack
            images = Variable(images).cuda() if torch.cuda.is_available() else Variable(images)
            gts = Variable(gts).cuda() if torch.cuda.is_available() else Variable(gts)

            with torch.no_grad():
                res = model(images)

                loss = criterion(res, gts)
                loss_record.update(loss.data, config.batch_size)

                res = torch.sigmoid(res)

                res = abs(res - res.min()) / (abs(res.max() - res.min()) + config.epsilon)

                inter = ((res * gts)).sum(dim=(2, 3))
                union = ((res + gts)).sum(dim=(2, 3))

                dice = (2 * abs(inter))/(abs(union) + config.epsilon)
                dice = float(dice.mean().data.cpu().numpy())

                val_score += dice

        loss_val.append(loss_record.show().data.cpu().numpy())
        val_score /= float(len(val_loader))
        scheduler.step(val_score)

        if not os.path.exists(train_save):
            os.makedirs(train_save)

        print("Best: ", best)
        print("Val: ", val_score)

        if val_score > best:
            best = val_score
            torch.save(model.state_dict(), os.path.join(train_save, f'GIVTEDNet_best.pth'))

        torch.save(model.state_dict(), os.path.join(train_save, f'GIVTEDNet.pth'))

        plt.plot(loss_train, color = 'r', label='train')
        plt.plot(loss_val, color = 'b', label='validation')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.grid()
        plt.savefig(f"./experiment/{config.dataset_name}/loss_plot.png")
        plt.clf()


        with open(f"./experiment/{config.dataset_name}/loss_history.txt", 'a') as f:
            f.write("Loss Train: [")
            for val in loss_train:
                f.write(f" {val} ")
            f.write("]\n")
            f.write("Loss Validation: [")
            for val in loss_val:
                f.write(f" {val} ")
            f.write("]\n")
            f.write("-----------------------------------------------------------\n")


        if early_stopping.early_stop(loss_val[-1]):
            print(f"Training stopped at epoch: {n}")
            break


if __name__ == "__main__":
    train_fn()
