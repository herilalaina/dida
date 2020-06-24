import glob
import shutil
import datetime
import pickle as pkl
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from dida_network import Net
from utils_sinkhorn import normalize_cov
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)

def train(model, criterion, optimizer, data_loader_train, plotter, epoch):
    global activations_in, activations_out

    model.train()
    list_loss = []

    nb_batch = len(data_loader_train)
    correct, N = 0, 0

    for idx, (X1, y1, info1, target) in enumerate(data_loader_train):
        optimizer.zero_grad()

        X1, y1, target = X1.to("cuda"), y1.to("cuda"), target.to("cuda")

        logits, _, _ = model(X1, y1, info1)

        loss = criterion(logits, target.long())
        loss.backward()

        pred = logits.max(1, keepdim=True)[1]
        correct_ = pred.eq(target.long().view_as(pred)).sum().item()
        correct += correct_

        list_loss.extend([loss.item()] * pred.size(0))
        N += target.size(0)

        print("-> {}: loss {} acc {}".format(idx, np.mean(list_loss), correct / N), flush=True)
        plotter.add_scalar("Step/loss", loss.item(), epoch * 2000 + idx)
        plotter.add_scalar("Step/acc", correct_ / pred.size(0), epoch * 2000 + idx)
        optimizer.step()

    return np.mean(list_loss), correct / N

def eval(model, criterion, data_loader_train):
    model.eval()
    list_loss = []
    list_acc = []
    correct, N = 0, 0

    with torch.no_grad():
        for idx, (X1, y1, info1, target) in enumerate(data_loader_train):
            X1, y1, target = X1.to("cuda"), y1.to("cuda"), target.to("cuda")
            logits, _, _ = model(X1, y1, info1)

            loss = criterion(logits, target.long())
            N += target.size(0)

            pred = logits.max(1, keepdim=True)[1]
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            list_loss.extend([loss.item()] * pred.size(0))


    return np.mean(list_loss), correct/N


def load_data(batch_size, npoints):
    data_train = OpenML_3D(npoints, eval_set="train")
    samples_weight = torch.tensor([1 / data_train.weights[t] for t in data_train.score])

    sampler = WeightedRandomSampler(samples_weight, 10000)

    data_loader_train = data.DataLoader(dataset=data_train,
                                                          drop_last=True,
                                                          collate_fn=collate_data_target,
                                                          batch_size=batch_size,
                                                          sampler=sampler)
    data_loader_valid = data.DataLoader(dataset=OpenML_3D(npoints, eval_set="valid"),
                                                         drop_last=False,
                                                         collate_fn=collate_data_target,
                                                         batch_size=batch_size,
                                                         shuffle=False)

    return data_loader_train, data_loader_valid


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    print("DEVICE USED ", device, flush=True)

    nb_binary_target = 2

    parameters = {
        "d_Mfeat": [4, 4],
        "d_Mlab": [1, 1],
        "d_out": [10, 1],
        "nmoments": [512, 1024],
        "tensorizations": [2, 2],
        "skip_connection": False,
        "use_batchnorm": True,
        "use_metafeatures": False,
        "fc_metafeatures": [256, 128, 64],
        "dropout_fc": [0.1, 0.1, 0.1]
    }

    parameters["lr"] = 1e-4
    parameters["batch_size"] = 32
    parameters["nb_epochs"] = 5000
    parameters["lr_scheduler"] = {"step": 10, "gamma": 0.75}
    parameters["N"] = 30
    parameters["npoints"] = 100
    parameters["npatchs"] = 10

    model = Net(d_feat=3, d_lab=1, npoints=parameters["npoints"], N=parameters["N"], parameters=parameters).to(device)

    checkpoint_dir = datetime.datetime.now().strftime('./checkpoints/log_%Y-%m-%d-%H-%M-%S-%f')
    os.mkdir(checkpoint_dir)
    for filename in glob.glob('*.py'):
        shutil.copy(filename, checkpoint_dir)

    writer = SummaryWriter(checkpoint_dir)

    data_loader_train, data_loader_val = load_data(parameters["batch_size"], parameters["npoints"])

    optimizer = optim.Adam(model.parameters(), lr=parameters["lr"], amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                parameters["lr_scheduler"]["step"],
                                                parameters["lr_scheduler"]["gamma"])

    del parameters["lr_scheduler"]
    writer.add_hparams({k: str(v) for k, v in parameters.items()}, {})

    criterion = torch.nn.NLLLoss(reduction='mean').to(device)
    best_loss = 99

    for epoch in range(1, parameters["nb_epochs"]):
        epoch_loss, epoch_acc = train(model, criterion, optimizer, data_loader_train, writer, epoch)
        scheduler.step()


        writer.add_scalar('Training/loss', epoch_loss, epoch)
        writer.add_scalar('Training/acc', epoch_acc, epoch)

        if epoch % 10 == 0:
            epoch_loss_train, epoch_acc_train = eval(model, criterion, data_loader_train)
            epoch_loss_valid, epoch_acc_valid = eval(model, criterion, data_loader_val)
            writer.add_scalar('Loss/train', epoch_loss_train, epoch)
            writer.add_scalar('Acc/train', epoch_acc_train, epoch)
            writer.add_scalar('Loss/valid', epoch_loss_valid, epoch)
            writer.add_scalar('Acc/valid', epoch_acc_valid, epoch)

        print("Epoch %s * loss train %s * " % (epoch, epoch_loss), flush=True)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_{}".format(epoch)))
