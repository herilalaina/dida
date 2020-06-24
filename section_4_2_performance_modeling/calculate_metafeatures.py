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

def eval(model, data_loader_train):
    model.eval()
    list_metafeatures = []

    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader_train), total=len(data_loader_train))
        for idx, (X1, y1, info1) in pbar:

            X1, y1, info1 = X1.to("cuda"), y1.to("cuda"), info1.to("cuda")

            _, _, list_z = model.extractor_sdn(X1, y1, info1)
            list_metafeatures.append(list_z[-4]) # Take output of last EB

    return torch.cat(list_metafeatures, dim=0)

def load_data(batch_size, npoints):

    loader = data.DataLoader(dataset=OpenML_3D_ALL(npoints, eval_set="train"),
                                                          collate_fn=collate_data_utils,
                                                          batch_size=batch_size,
                                                          shuffle=False)

    return loader


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
    model.load_state_dict(torch.load("checkpoints/investigate_test_2020-05-22-01-04-39-336754/model_106"))

    data_loader = load_data(parameters["batch_size"], parameters["npoints"])

    metafeatures = eval(model, data_loader)
    print(metafeatures.size())
    np.save("results/cluster_metafeatures", metafeatures.data.cpu().numpy())
