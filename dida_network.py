import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


from block import EB_variable, EB


class Extractor(nn.Module):
    def __init__(self, d_Mfeat, d_Mlab, npoints, N, nmoments, d_out, skip_connection, use_batchnorm, fc_metafeatures, tensorizations, dropout_fc):
        super(Extractor, self).__init__()
        self.N = N
        self.d_out = d_out
        self.d_Mlab = d_Mlab
        self.d_Mfeat = d_Mfeat
        self.npoints = npoints
        self.nmoments = nmoments
        self.skip_connection = skip_connection
        self.use_batchnorm = use_batchnorm
        self.tensorizations = tensorizations
        self.fc_metafeatures = fc_metafeatures
        self.dropout_fc = dropout_fc

        self.previous_nmoments = self.nmoments.copy()

        for i in range(len(self.nmoments)):
            if i == 0:
                self.previous_nmoments[i] = 1
            else:
                self.previous_nmoments[i] = self.nmoments[(i-1)] + (self.previous_nmoments[i-1] if self.skip_connection else 0)

        self.list_module = torch.nn.ModuleList()
        self.list_batch_norm = torch.nn.ModuleList()
        self.list_dropout_z = torch.nn.ModuleList()
        self.list_dropout_x = torch.nn.ModuleList()

        for i, (d_Ml, d_Mf, d_o, n_moment) in enumerate(zip(self.d_Mlab, self.d_Mfeat,
                                                            self.d_out, self.nmoments)):
            if i == 0:
                self.list_module.append(EB_variable(d_feat=self.d_out[i-1],
                                                                    d_Mfeat=d_Mf,
                                                                    d_out=d_o,
                                                                    N=self.N,
                                                                    npoints=self.npoints,
                                                                    nmoments=n_moment,
                                                                    previous_nmoments=self.previous_nmoments[i],
                                                                    orders=[self.tensorizations[i]],
                                                                    position=0 if (i==0) else i,
                                                                    with_label=True))
            else:
                self.list_module.append(EB(d_feat=self.d_out[i-1],
                                                                    d_Mfeat=d_Mf,
                                                                    d_out=d_o,
                                                                    N=self.N,
                                                                    npoints=self.npoints,
                                                                    nmoments=n_moment,
                                                                    previous_nmoments=self.previous_nmoments[i],
                                                                    orders=[self.tensorizations[i]],
                                                                    position= -1 if i == len(self.nmoments) -1 else i
                                                                    ))
            if self.use_batchnorm:
                self.list_batch_norm.append(
                    nn.BatchNorm1d(n_moment + (self.previous_nmoments[i-1] if self.skip_connection else 0), momentum=0.1)) #

            self.list_dropout_z.append(nn.Dropout(0.01))
            self.list_dropout_x.append(nn.Dropout(0.01))

        self.list_fc = torch.nn.ModuleList()
        self.list_bn_fc = torch.nn.ModuleList()
        self.list_dropout_fc = torch.nn.ModuleList()
        for i, dim in enumerate(self.fc_metafeatures):
            if i == 0:
                self.list_fc.append(nn.Linear(self.nmoments[-1], dim))
            else:
                self.list_fc.append(nn.Linear(self.fc_metafeatures[i-1], dim))

            if i != len(self.fc_metafeatures) - 1:
                self.list_bn_fc.append(nn.BatchNorm1d(dim))
                self.list_dropout_fc.append(nn.Dropout(self.dropout_fc[i]))



    def forward(self, x, lab, info):
        batch_size = x.size(0)
        list_z = []

        for i in range(len(self.list_module)):
            if i == 0:
                x, z_out, min_npoints = self.list_module[i](x, lab, torch.zeros(x.size(0), 1).to("cuda"), info)
                del info, lab
            else:
                x, z_out = self.list_module[i](x, z)

            z = torch.cat([z, z_out], dim=1) if self.skip_connection and i > 0 else z_out
            z = self.list_batch_norm[i](z.unsqueeze(-1)).squeeze(-1) if self.use_batchnorm else z
            z = self.list_dropout_z[i](z)

            list_z.append(z)

        for i in range(len(self.list_fc)):
            z = self.list_fc[i](z)

            if i != len(self.list_fc) - 1:
                if self.use_batchnorm:
                    z = self.list_bn_fc[i](z)
                z = F.relu(z)
                z = self.list_dropout_fc[i](z)
            list_z.append(z)

        return z, x, list_z


class Net(nn.Module):
    def __init__(self, d_feat, d_lab, npoints, N, parameters):
        super(Net, self).__init__()
        self.npoints = npoints
        self.N = N

        self.d_Mfeat = parameters["d_Mfeat"]
        self.d_Mlab = parameters["d_Mlab"]
        self.nmoments = parameters["nmoments"]
        self.d_out = parameters["d_out"]
        self.skip_connection = parameters["skip_connection"]
        self.use_batchnorm = parameters["use_batchnorm"]
        self.use_metafeatures = parameters["use_metafeatures"]
        self.final_fc_output_dim = parameters["final_fc_output_dim"]
        self.fc_metafeatures = parameters["fc_metafeatures"]
        self.nb_output_class = parameters["nb_output_class"]
        self.tensorizations = parameters["tensorizations"]
        self.dropout_fc = parameters["dropout_fc"]

        self.fc1 = nn.Linear(self.fc_metafeatures[-1], self.final_fc_output_dim)
        self.extractor_sdn = Extractor(d_Mfeat=self.d_Mfeat,
                                        d_Mlab=self.d_Mlab,
                                        npoints=self.npoints,
                                        N=30,
                                        nmoments=self.nmoments,
                                        d_out=self.d_out,
                                        skip_connection=self.skip_connection,
                                        use_batchnorm=self.use_batchnorm,
                                        fc_metafeatures=self.fc_metafeatures,
                                        tensorizations=self.tensorizations,
                                        dropout_fc=self.dropout_fc)


    def forward(self, x1, lab1, info):
        z_out, x, list_z = self.extractor_sdn(x1, lab1, info)
        z = self.fc1(z_out)
        return F.log_softmax(z.view(z.size(0),
                                    self.nb_output_class,
                                    2), dim=2), z_out, list_z



class BatchIdentNet(nn.Module):
    def __init__(self, d_feat, d_lab, npoints, N, parameters):
        super(BatchIdentNet, self).__init__()
        self.npoints = npoints
        self.N = N

        self.d_Mfeat = parameters["d_Mfeat"]
        self.d_Mlab = parameters["d_Mlab"]
        self.nmoments = parameters["nmoments"]
        self.d_out = parameters["d_out"]
        self.skip_connection = parameters["skip_connection"]
        self.use_batchnorm = parameters["use_batchnorm"]
        self.use_metafeatures = parameters["use_metafeatures"]
        self.fc_metafeatures = parameters["fc_metafeatures"]
        self.nb_output_class = parameters["nb_output_class"]
        self.tensorizations = parameters["tensorizations"]
        self.dropout_fc = parameters["dropout_fc"]

        self.extractor_sdn = Extractor(d_Mfeat=self.d_Mfeat,
                                        d_Mlab=self.d_Mlab,
                                        npoints=self.npoints,
                                        N=30,
                                        nmoments=self.nmoments,
                                        d_out=self.d_out,
                                        skip_connection=self.skip_connection,
                                        use_batchnorm=self.use_batchnorm,
                                        fc_metafeatures=self.fc_metafeatures,
                                        tensorizations=self.tensorizations,
                                        dropout_fc=self.dropout_fc)


    def forward(self, x1, lab1, info1, x2, lab2, info2, train=True):
        if train:
            self.extractor_sdn.train()
            z_1, _, _ = self.extractor_sdn(x1, lab1, info1)
            self.extractor_sdn.eval() # Because of batchnorm in siamese network
            z_2, _, _ = self.extractor_sdn(x2, lab2, info2)
        else:
            z_1, _, _ = self.extractor_sdn(x1, lab1, info1)
            z_2, _, _ = self.extractor_sdn(x2, lab2, info2)
        diff = torch.abs(z_1 - z_2)
        z = torch.exp(- torch.norm(diff, 2, dim=1, keepdim=True))
        return torch.cat((z, 1 - z), 1)
