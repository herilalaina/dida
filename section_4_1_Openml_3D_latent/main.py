import glob
import shutil
import datetime
import collections
import pickle as pkl
from functools import partial
from collections import OrderedDict, defaultdict

from collate import collate_data
from dataloader_latent import DATA_OpenML_3D

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import matplotlib
matplotlib.use('Agg')

from utils_sinkhorn import normalize_cov

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from dida_network import BatchIdentNet
from torch.utils.tensorboard import SummaryWriter



def train(model, criterion, optimizer, data_loader_train, plotter, epoch):
    global activations_in, activations_out

    model.train()
    list_loss = []
    list_acc = []

    nb_batch = len(data_loader_train)
    mean_loss, mean_acc = 0, 0

    pbar = tqdm(enumerate(data_loader_train), total=len(data_loader_train))
    for idx, (X1, y1, info1, X2, y2, info2, target) in pbar:
        optimizer.zero_grad()

        X1, y1, target = X1.to("cuda"), y1.to("cuda"), target.to("cuda")
        X2, y2 = X2.to("cuda"), y2.to("cuda")

        logits = model(X1, y1, info1, X2, y2, info2)

        loss = criterion(logits, target.long())
        loss.backward()

        pred = logits.max(1, keepdim=True)[1]
        correct = pred.eq(target.long().view_as(pred)).sum().item() / pred.size(0)

        mean_loss = (mean_loss * idx + loss.item()) /  (idx + 1)
        mean_acc = (mean_acc * idx + correct) /  (idx + 1)

        pbar.set_description("iter {}: \t loss: {}\t accuracy: {}".format(idx, mean_loss, mean_acc))

        list_loss.append(loss.item())
        list_acc.append(correct)

        optimizer.step()

    return np.mean(list_loss[-idx:]), np.mean(list_acc[-idx:])

def eval(model, criterion, data_loader_train):
    model.eval()
    list_loss = []
    list_acc = []

    list_layer = {}


    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader_train), total=len(data_loader_train))
        for idx, (X1, y1, info1, X2, y2, info2, target) in pbar:

            X1, y1, target = X1.to("cuda"), y1.to("cuda"), target.to("cuda")
            X2, y2 = X2.to("cuda"), y2.to("cuda")

            logits = model(X1, y1, info1, X2, y2, info2)

            loss = criterion(logits, target.long())

            pred = logits.max(1, keepdim=True)[1]
            correct = pred.eq(target.long().view_as(pred)).sum().item() / pred.size(0)

            list_loss.append(loss.item())
            list_acc.append(correct)

    return np.mean(list_loss[-idx:]), np.mean(list_acc[-idx:]), list_layer

def load_data(batch_size, npoints, npatchs):

    data_loader_train = data.DataLoader(dataset=DATA_OpenML_3D(npoints, eval_set="train", npatchs=npatchs),
                                                          drop_last=True,
                                                          collate_fn=collate_data,
                                                          batch_size=batch_size,
                                                          shuffle=True)
    data_loader_valid = data.DataLoader(dataset=DATA_OpenML_3D(npoints, eval_set="valid", npatchs=npatchs),
                                                         drop_last=False,
                                                         collate_fn=collate_data,
                                                         batch_size=batch_size,
                                                         shuffle=False)

    return data_loader_train, data_loader_valid


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    print("DEVICE USED ", device)

    nb_binary_target = 2

    parameters = {
        "d_Mfeat": [8, 8],
        "d_Mlab": [1, 1],
        "d_out": [10, 3],
        "nmoments": [512, 1024],
        "tensorizations": [4, 4],
        "final_fc_output_dim": nb_binary_target * 2,
        "skip_connection": False,
        "use_batchnorm": True,
        "use_metafeatures": False,
        "fc_metafeatures": [256, 128, 64],
        "dropout_fc": [0.01, 0.01, 0.01],
        "nb_output_class": nb_binary_target
    }

    parameters["lr"] = 1e-4
    parameters["batch_size"] = 32
    parameters["nb_epochs"] = 5000
    parameters["lr_scheduler"] = {"step": 10, "gamma": 0.75}
    parameters["N"] = 30
    parameters["npoints"] = 100
    parameters["npatchs"] = 10

    model = BatchIdentNet(d_feat=3, d_lab=1, npoints=parameters["npoints"], N=parameters["N"], parameters=parameters).to(device)

    checkpoint_dir = datetime.datetime.now().strftime('./checkpoints/latent_%Y-%m-%d-%H-%M-%S-%f')
    os.mkdir(checkpoint_dir)
    for filename in glob.glob('*.py'):
        shutil.copy(filename, checkpoint_dir)

    writer = SummaryWriter(checkpoint_dir)

    data_loader_train, data_loader_val = load_data(parameters["batch_size"], parameters["npoints"], parameters["npatchs"])

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

        epoch_loss_train, epoch_acc_train, list_feat = eval(model, criterion, data_loader_train)
        epoch_loss_valid, epoch_acc_valid, list_feat = eval(model, criterion, data_loader_val)

        writer.add_scalar('Loss/training', epoch_loss, epoch)
        writer.add_scalar('Loss/valid', epoch_loss_valid, epoch)
        writer.add_scalar('Loss/train', epoch_loss_train, epoch)
        writer.add_scalar('Acc/training', epoch_acc, epoch)
        writer.add_scalar('Acc/valid', epoch_acc_valid, epoch)
        writer.add_scalar('Acc/train', epoch_acc_train, epoch)

        print("Epoch %s\t loss train %s\t loss valid %s\t acc train %s\t acc valid %s" % (epoch, epoch_loss, epoch_loss_valid,
                                                                                            epoch_acc, epoch_acc_valid))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_{}".format(epoch)))
