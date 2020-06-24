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

from dida_network import BatchIdentNet
from utils_sinkhorn import normalize_cov
from torch.utils.tensorboard import SummaryWriter


class UCI_data(data.Dataset):
    """ UCI Dataset.
    """

    def __init__(self, set_data):
        name_dataset = os.path.join("/linkhome/rech/genini01/uvp29is/Code/metanal/datasets/uci/", set_data)

        target = np.load(os.path.join(name_dataset, "target.npy"))
        X = pkl.load(open(os.path.join(name_dataset, "X.npy"), "rb"))
        Y = np.load(os.path.join(name_dataset, "y.npy"), allow_pickle=True)
        self.len = len(target)

        self.X = [[torch.from_numpy(x) for x in X_] for X_ in X]
        self.y = [[torch.from_numpy(y) for y in Y_] for Y_ in Y]
        self.target = torch.from_numpy(target)

    def __getitem__(self, index):
        t1, t2 = self.target[index]
        X1 = torch.cat([self.X[t1[0]][t1[i]] for i in range(1, 2)])
        Y1 = torch.cat([torch.argmax(self.y[t1[0]][t1[i]], dim=1) for i in range(1, 8)])

        X2 = torch.cat([self.X[t2[0]][t2[i]] for i in range(1, 2)])
        Y2 = torch.cat([torch.argmax(self.y[t2[0]][t2[i]], dim=1) for i in range(1, 8)])

        return X1, Y1, X2, Y2, 1 - (t1[0] == t2[0]).int()

    def __len__(self):
        return self.len

def collate_data(batch):
    target = torch.stack([item[4] for item in batch])
    x1, lab1, info1 = collate_data_utils([[item[0], item[1]] for item in batch])
    x2, lab2, info2 = collate_data_utils([[item[2], item[3]] for item in batch])
    return x1, lab1, info1, x2, lab2, info2, target

def collate_data_utils(batch):
    size_x1 = [item[0].size(0) * item[0].size(1) for item in batch]
    shape_x1 = [(item[0].size(0), item[0].size(1)) for item in batch]

    x1 = []
    lab1 = []
    target = []
    info = []

    for i, item in enumerate(batch):
        pad_x1 = torch.zeros(size=(int(np.max(size_x1)), ))
        pad_x1[:size_x1[i]] = item[0].view(size_x1[i])

        pad_lab1 = torch.zeros(size=(int(np.max([v[0] for v in shape_x1])), ))
        pad_lab1[:item[1].size(0)] = item[1]

        x1.append(pad_x1)
        lab1.append(pad_lab1)
        info.append([size_x1[i], shape_x1[i][0]])

    x1 = torch.stack(x1)
    lab1 = torch.stack(lab1)
    info = torch.LongTensor(info)

    return x1, lab1, info



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

            logits = model(X1, y1, info1, X2, y2, info2, train=False)

            loss = criterion(logits, target.long())

            pred = logits.max(1, keepdim=True)[1]
            correct = pred.eq(target.long().view_as(pred)).sum().item() / pred.size(0)

            list_loss.append(loss.item())
            list_acc.append(correct)

    return np.mean(list_loss), np.mean(list_acc), list_layer

def load_data(batch_size):

    data_loader_train = data.DataLoader(dataset=UCI_data(set_data="train"),
                                                          drop_last=True,
                                                          collate_fn=collate_data,
                                                          batch_size=batch_size,
                                                          shuffle=True)
    data_loader_test = data.DataLoader(dataset=UCI_data(set_data="test"),
                                                         drop_last=False,
                                                         collate_fn=collate_data,
                                                         batch_size=batch_size,
                                                         shuffle=False)

    return data_loader_train, data_loader_test


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    print("DEVICE USED ", device)

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
        "dropout_fc": [0.1, 0.1, 0.1],
    }

    parameters["lr"] = 1e-4
    parameters["batch_size"] = 32
    parameters["nb_epochs"] = 1000
    parameters["lr_scheduler"] = {"step": 10, "gamma": 0.75}
    parameters["N"] = 30
    parameters["npoints"] = 100
    parameters["npatchs"] = 10

    model = BatchIdentNet(d_feat=3, d_lab=1, npoints=parameters["npoints"], N=parameters["N"], parameters=parameters).to(device)

    checkpoint_dir = datetime.datetime.now().strftime('./checkpoints/log_%Y-%m-%d-%H-%M-%S-%f')
    os.mkdir(checkpoint_dir)
    for filename in glob.glob('*.py'):
        shutil.copy(filename, checkpoint_dir)

    writer = SummaryWriter(checkpoint_dir)

    data_loader_train, data_loader_val = load_data(parameters["batch_size"])

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
