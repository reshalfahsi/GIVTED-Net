from torch.autograd import Variable
from datetime import datetime

from tools.utils import AvgMeter


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer, 
    n, 
    epoch,
    batch_size, 
    total_step,
):
    model.train()

    loss_record = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):

        optimizer.zero_grad()

        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # ---- forward ----
        pred = model(images)

        # ---- loss function ----
        loss = criterion(pred, gts)

        # ---- backward ----
        loss.backward()

        # clip_gradient(optimizer, clip)
        optimizer.step()

        # ---- recording loss ----
        loss_record.update(loss.data, batch_size)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{}/{}], Step [{}/{}], '
                  '[loss: {}'
                  ', lr: {}'
                  ']'.
                  format(
                      datetime.now(),
                      n, epoch, i, total_step,
                      loss_record.show(), optimizer.param_groups[0]['lr'],
                  )
            )

    return loss_record.show()
