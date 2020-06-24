
import os
import sys
import time
import glob
import torch
import random
import argparse

import numpy as np

import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from intrinsic import intrinsicEstimator
from torchvision.utils import save_image
from torchvision import datasets, transforms
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--id', type=int, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# SEED
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
np.random.seed(args.seed)  # Numpy module.
random.seed(args.seed)  # Python random module.
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

dimension = 10


class Data(torch.utils.data.Dataset):
    """ Point Cloud Dataset.
    """
    def __init__(self, X, y):
        self.len = len(X)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)


    def __getitem__(self, index):
        return self.X[index].float(), self.y[index].float()

    def __len__(self):
        return self.len

    def _init_fn(worker_id):
        np.random.seed(int(args.seed))


class autoencoder(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_latent):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden_1),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_latent))

        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden_1),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_input))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_data(path):
    X = np.load(path)
    feat = np.load(path.replace(".x.npy", ".feat.npy"))
    print("list features", feat)

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [i for i in range(len(feat)) if feat[i] == 1])],
                                            remainder='passthrough', n_jobs=-1, sparse_threshold=0)
    X = np.array(columnTransformer.fit_transform(X))
    y = np.load(path.replace(".x.npy", ".y.npy"))
    assert len(X) == len(y)

    if X.shape[0] > 50000:
        to_select = np.random.randint(0, X.shape[0], 50000)
        X = X[to_select]
        y = y[to_select]

    if X.shape[1] > 20000:
        X = SelectKBest(chi2, k=1000).fit_transform(X, y)


    intrinsic_value = int(dimension)

    return Data(X, y), intrinsic_value, X.shape[1]

def save_data(train_loader, model, epoch, path):
    result_data = []
    y = []
    model.eval()
    with torch.no_grad():
        for data, y_ in train_loader:
            data, y_ = data.to(device), y_.to(device)
            output = model.encoder(data).data.cpu().numpy()
            y.extend(y_.data.cpu().numpy())
            result_data.extend(output)

    np.save(path.replace("clean_data", "version_{}D".format(dimension)).replace(".x.npy", "_epoch{}.x.npy".format(epoch)) , np.array(result_data))
    np.save(path.replace("clean_data", "version_{}D".format(dimension)).replace(".x.npy", "_epoch{}.y.npy".format(epoch)) , np.array(y))

def get_code(path):
    print("Preprocess data", path)

    data_to_load, n_latent, n_input = get_data(path)

    n_hidden_1 = int((n_input + n_latent) / 2)
    print("n_latent: {0}\t n_hidden_1: {1}\t n_input: {2}".format(n_latent, n_hidden_1, n_input))

    model = autoencoder(n_input, n_hidden_1, n_latent).to(device)
    # model = torch.nn.DataParallel(model).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_loader = torch.utils.data.DataLoader(data_to_load, batch_size=args.batch_size, shuffle=True, **kwargs)
    save_loader = torch.utils.data.DataLoader(data_to_load, batch_size=args.batch_size, shuffle=False, **kwargs)


    counter = 0
    best_loss = 1000

    for epoch in range(args.epochs):
        model.train()
        list_loss = []
        for data, y_ in train_loader:
            data, y_ = data.to(device), y_.to(device)
            # ===================forward=====================
            output = model(data)
            loss = criterion(output, data)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            list_loss.append(loss.item())

        loss_value = np.mean(list_loss)

        if best_loss <= loss_value:
            counter += 1
        else:
            best_loss = loss_value
            counter = 0

        if epoch % 10 == 0:
            save_data(save_loader, model, epoch, path)

        if counter >= 20:
            break

        print("Epoch {}: loss: {} counter: {}".format(epoch + 1, loss_value, counter))





if __name__ == "__main__":
    # OpenML dataset ID
    list_files = ['1475_0', '4135_9', '1508_0', '1465_0', '793_0', '1557_1', '42016_0', '1540_0', '796_1', '41945_0', '974_0', '1015_0', '768_0', '792_0', '1538_0', '300_0', '1115_4', '811_1', '679_0', '740_0', '930_1', '467_0', '995_0', '893_0', '465_3', '882_0', '1491_0', '1006_15', '42046_0', '46_60', '38_22', '1510_0', '314_22', '311_0', '727_0', '758_0', '40709_2', '943_0', '40669_6', '1049_0', '346_0', '1524_0', '1413_0', '726_0', '1542_0', '1530_0', '749_0', '1061_0', '803_0', '919_0', '185_1', '40685_0', '181_0', '1463_5', '1490_0', '761_0', '41674_2', '878_0', '5_73', '807_0', '1513_0', '161_0', '40711_7', '54_0', '40704_0', '62_15', '40690_9', '874_0', '41679_2', '855_0', '1497_0', '935_0', '903_0', '41156_0', '41146_0', '60_0', '450_1', '459_0', '40498_0', '841_0', '42041_0', '734_0', '2_32', '61_0', '784_1', '1493_0', '942_1', '1067_0', '815_0', '1554_4', '151_1', '1488_0', '1063_0', '723_0', '959_8', '1046_0', '40650_20', '15_0', '808_0', '40496_0', '1512_0', '991_6', '1055_1', '988_1', '336_0', '756_0', '41946_0', '1528_0', '921_1', '719_4', '4534_30', '375_0', '1482_0', '35_33', '40678_7', '979_0', '24_22', '1121_3', '1444_0', '973_0', '1026_6', '1467_0', '1120_0', '795_0', '951_1', '934_4', '880_0', '949_1', '1167_1', '1056_0', '1527_0', '179_12', '31_13', '28_0', '37_0', '1073_0', '53_0', '889_0', '444_3', '40707_23', '860_0', '724_1', '184_6', '188_5', '987_2', '938_1', '762_0', '871_0', '1002_39', '21_6', '912_0', '917_0', '448_3', '1044_3', '1449_0', '774_0', '23499_9', '966_1', '1505_1', '267_0', '962_0', '783_0', '41228_0', '40708_23', '255_7', '827_0', '767_2', '964_1', '847_0', '843_0', '180_40', '40985_0', '40984_0', '750_0', '1556_5', '357_0', '1009_4', '40713_23', '1503_0', '34_8', '57_22', '1480_1', '472_0', '752_0', '40647_20', '780_0', '40648_20', '894_0', '342_3', '731_0', '785_0', '40994_0', '865_1', '4329_13', '40682_0', '1529_0', '1489_0', '1541_0', '12_0', '42036_0', '1568_8', '265_0', '925_0', '838_0', '162_0', '1075_0', '813_0', '772_0', '40701_4', '1455_5', '351_0', '335_0', '818_1', '1495_6', '829_0', '1450_0', '40981_8', '1560_0', '730_0', '40705_2', '1453_0', '862_2', '29_9', '1211_0', '42051_0', '1492_0', '4154_0', '1446_0', '1100_4', '833_0', '799_0', '41684_2', '751_0', '826_11', '49_7', '40660_11', '790_1', '40476_20', '42066_0', '733_0', '1555_3', '1483_2', '40686_12', '923_1', '464_0', '18_0', '897_2', '1441_0', '1000_22', '895_0', '763_0', '480_8', '40922_0', '1466_0', '41027_0', '754_0', '1543_0', '457_0', '285_3', '1219_0', '755_0', '40681_6', '1494_0', '835_4', '461_3', '846_0', '1048_0', '1499_0', '14_0', '41005_26', '1059_0', '1013_1', '40663_20', '955_2', '777_0', '26_8', '745_1', '42186_0', '717_0', '804_4', '479_3', '153_0', '928_1', '42071_0', '1459_0', '929_0', '1506_13', '822_0', '836_3', '469_4', '725_0', '735_0', '42021_0', '310_0', '1498_1', '1511_0', '1443_0', '1539_0', '775_0', '714_2', '187_0', '1552_4', '41007_26', '554_0', '13_9', '42_35', '41997_0', '23_7', '728_0', '927_0', '782_0', '764_2', '1016_2', '800_0', '1053_0', '729_0', '1597_0', '1487_0', '744_0', '40700_1', '41680_2', '1068_0', '950_1', '881_3', '6_0', '879_0', '1062_0', '916_0', '40499_0', '997_0', '1218_0', '1547_0', '776_0', '4538_0', '823_0', '869_0', '20_240', '868_0', '42098_0', '1523_0', '892_0', '884_0', '1018_41', '1532_0', '924_0', '40714_5', '468_0', '1600_0', '476_0', '1500_0', '40649_20', '4_8', '42056_0', '778_0', '913_0', '1544_0', '773_0', '1451_0', '1050_0', '1565_0', '845_0', '1526_0', '43_1', '55_13', '40668_42', '1504_0', '1460_0', '40475_20', '44_0', '41004_26', '1041_0', '339_1', '683_0', '40671_0', '1005_0', '820_0', '152_0', '340_3', '40497_0', '887_1', '1025_5', '1549_3', '891_2', '941_7', '3_36', '40999_26', '59_0', '36_0', '337_0', '307_2', '901_0', '40683_8', '42011_0', '896_0', '830_0', '870_0', '721_0', '787_0', '22_0', '753_0', '857_0', '715_0', '16_0', '789_0', '817_0', '1551_3', '40982_0', '1473_0', '1536_0', '40478_20', '41682_2', '1501_0', '30_0', '771_2', '182_0', '1060_0', '354_0', '42193_1', '41950_0', '40687_12', '812_0', '1014_4', '39_0', '40677_24', '1021_0', '41511_0', '748_2', '137_9', '926_0', '40710_8', '1496_0', '791_0', '1471_0', '1412_0', '259_0', '463_26', '1509_0', '1180_0', '1118_6', '1525_0', '1448_0', '952_0', '993_27', '933_0', '48_2', '902_4', '40693_9', '947_1', '40900_0', '908_0', '56_16', '41568_0', '41583_0', '1071_0', '42192_1', '40975_6', '40477_20', '746_0', '824_0', '41675_2', '1464_0', '23512_0', '1553_4', '41998_2', '40474_20', '40983_0', '1507_0', '890_1', '333_0', '848_1', '1116_1', '1012_26', '965_15', '40646_20', '694_0', '1570_0', '743_0', '41521_2', '42003_0', '910_0', '1011_0', '969_0', '41538_3', '936_0', '40691_0', '329_0', '900_0', '1217_0', '914_0', '40706_10', '819_0', '859_0', '41000_26', '251_0', '1117_0', '338_6', '1534_0', '119_9', '345_44', '1054_0', '42031_0', '42026_0', '915_3', '905_0', '9_10', '909_0', '931_0', '40664_0', '885_0', '682_0', '1559_0', '911_0', '1040_0', '759_0', '1533_0', '853_1', '736_0', '1442_0', '40998_26', '1567_0', '1535_0', '1545_0', '906_0', '1220_0', '40997_26', '886_0', '1019_0', '164_0', '976_0', '770_0', '816_0', '983_7', '958_0', '1065_0', '334_0', '1216_0', '737_0', '832_0', '40702_10', '40971_0', '446_0', '1537_0', '747_4', '1064_0', '825_3', '51_7', '1452_0', '41685_2', '41919_0', '40680_10', '475_0', '821_0', '23517_0', '849_0', '1069_0', '1481_3', '1462_0', '907_0', '41671_0', '1546_0', '875_3', '1036_0', '685_0', '462_1', '41_0', '50_9', '794_0', '11_0', '32_0', '10_15', '945_3', '996_0', '1461_9', '41544_2', '801_0', '814_0', '765_2', '40_0', '864_1', '741_1', '946_0', '977_0', '1558_9', '779_0', '1502_0', '863_0', '994_0', '1447_0', '722_0', '720_1', '713_0', '1531_0', '867_2']
    id = args.id

    f = "/linkhome/rech/genini01/uvp29is/Code/metanal/datasets/openml/clean_data/{}.x.npy".format(list_files[id])
    try:
        get_code(f)
    except Exception as e:
        raise(e)
