
import os
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from utils_sinkhorn import *

class EB(nn.Module):
    """ Pairwise interactions block.
    """

    def __init__(self, d_feat, d_Mfeat, d_out, N, npoints, nmoments, previous_nmoments, orders, position, with_label = False):
        super(EB, self).__init__()
        self.d_feat = d_feat
        self.npoints = npoints
        self.d_Mfeat = d_Mfeat  # feature moments -- u
        self.d_out = d_out  # output feature dimension
        self.N = N
        self.nmoments = nmoments
        self.orders = orders
        self.with_label = with_label
        self.position = position

        self.list_meas_feat = torch.nn.ModuleList()
        self.list_vect_feat = torch.nn.ModuleList()

        for o in self.orders:
            if self.position != -1:
                self.list_meas_feat.append(nn.Linear(o, self.d_Mfeat))
            self.list_vect_feat.append(nn.Linear(o, self.d_Mfeat))


        if self.with_label:
            self.d_Mlab = 1  # feature moments -- u
            if self.position != -1:
                self.meas_x = nn.Linear(self.d_Mfeat * len(self.orders) + self.d_Mlab, self.d_out)
            self.vect_x = nn.Linear(self.d_Mfeat * len(self.orders) + self.d_Mlab, self.nmoments)
            self.vect_mlab_pos = torch.nn.Parameter(torch.randn((1)))
            self.vect_mlab_neg = torch.nn.Parameter(torch.randn((1)))
            self.meas_mlab_pos = torch.nn.Parameter(torch.randn((1)))
            self.meas_mlab_neg = torch.nn.Parameter(torch.randn((1)))
        else:
            if self.position != -1:
                self.meas_x = nn.Linear(self.d_Mfeat * len(self.orders), self.d_out)
            self.vect_x = nn.Linear(self.d_Mfeat * len(self.orders), self.nmoments)
            self.vect_mlab_pos = None
            self.vect_mlab_neg = None
            self.meas_mlab_pos = None
            self.meas_mlab_neg = None

        # self.bn_meas = nn.BatchNorm1d(self.d_out, momentum=0.1)
        # self.bn_vect = nn.BatchNorm1d(self.nmoments, momentum=0.1)

        self.first = True if self.position == 0 else False
        self.last = True if self.position == -1 else False

        if self.first:
            self.meas_z = nn.Linear(1, self.d_out)
            self.vect_z = nn.Linear(1, self.nmoments)
        else:
            self.meas_z = nn.Linear(previous_nmoments, self.d_out)
            self.vect_z = nn.Linear(previous_nmoments, self.nmoments)

    def forward(self, x, z, labels=None):
        # compute pairwise distances for nearest neighbor search.
        d_feat = self.d_feat
        batch_size = x.size(0)
        npoints = int(x.size(1) / d_feat)


        N = self.N if npoints > self.N else int(npoints / 2)
        # print(N, npoints, self.npoints)

        # print(x.size())
        distances = torch.sqrt(batch_Lpcost(x, x, 2, d_feat))
        distances[distances == 0] = 99999999
        # distances.masked_fill_(torch.eye(npoints, npoints).unsqueeze(0).repeat(batch_size, 1, 1).bool(), 0)

        # select N nonzero interactions of interest per point.
        val, idx = torch.topk(distances, N, 2, largest=False, sorted=True)
        distances = None
        val = None

        # tensorized features of size (batch_size,(N-1)*npoints,2*d_feat)
        # print("####", x.view(batch_size,npoints,d_feat).size(), idx.size())
        list_m_feat, list_v_feat = [], []

        moments = None #torch.zeros(batch_size, (N-1) * npoints, self.d_Mfeat).to("cuda")
        v_moments = None # torch.zeros(batch_size, (N-1) * npoints, self.d_Mfeat).to("cuda")

        size_moment = 0
        for i, order in enumerate(self.orders):
            to_select = torch.FloatTensor([[0] + list(l) for l in list(itertools.combinations(range(1, N), order-1))[:(N-1)]]).long()
            # print("order", i, to_select.size())
            x_ = torch.cat([
                    torch.gather(x.view(batch_size, npoints, d_feat).unsqueeze(1).repeat(1, npoints, 1, 1),
                    2,
                    idx[:, :, t].unsqueeze(3).repeat(1, 1, 1, d_feat)).unsqueeze(2)
                    for t in to_select], dim=2).view(batch_size, (N-1) * npoints, order * d_feat)

            x_ = torch.cat(torch.chunk(x_.view(batch_size, order*(N-1) * npoints, d_feat), d_feat,dim=2), 1).squeeze(2).view(batch_size,(N-1) * npoints* d_feat, order)

            if not self.last: # Do not compute for last layer
                m_feat = self.list_meas_feat[i](x_)
                m_feat = F.relu(m_feat)
                m_feat = torch.mean(torch.stack(
                    torch.chunk(m_feat.view(batch_size,
                                            (N - 1) * npoints * self.d_feat, self.d_Mfeat),
                                self.d_feat, dim=1), 1), 1)

            v_feat = self.list_vect_feat[i](x_)
            v_feat = F.relu(v_feat)
            v_feat = torch.mean(torch.stack(
                torch.chunk(v_feat.view(batch_size,
                                        (N - 1) * npoints * self.d_feat, self.d_Mfeat),
                            self.d_feat, dim=1), 1), 1)


            if v_moments is None:
                moments = m_feat if not self.last else None
                v_moments = v_feat
            else:
                moments = torch.cat([moments, m_feat], dim=2) if not self.last else None
                v_moments = torch.cat([v_moments, v_feat], dim=2)

            x_ = None
            m_feat = None
            v_feat = None

        if self.with_label:
            labels = batch_index_select_NN(labels.view(batch_size,
                                                        npoints, 1), idx)

            m_lab = torch.mean((labels[:, :, :1] == labels[:, :, 1:]).float(), 2) * self.meas_mlab_pos + \
                                torch.mean((labels[:, :, :1] != labels[:, :, 1:]).float(), 2) * self.meas_mlab_neg
            m_lab = m_lab.unsqueeze(2)
            v_lab = torch.mean((labels[:, :, :1] == labels[:, :, 1:]).float(), 2) * self.vect_mlab_pos + \
                                torch.mean((labels[:, :, :1] != labels[:, :, 1:]).float(), 2) * self.vect_mlab_neg
            v_lab = v_lab.unsqueeze(2)

            m_lab = F.relu(m_lab)
            v_lab = F.relu(v_lab)


            moments = torch.cat([moments, m_lab], 2) # size (batch_size,(N-1)*npoints,Mfeat+Mlab)
            v_moments = torch.cat([v_moments, v_lab], 2) # size (batch_size,(N-1)*npoints,Mfeat+Mlab)

            m_lab, v_lab = None, None

        # apply final layer to moments -- output measure
        if self.last:
            x_new = None
        else:
            x_new = self.meas_x(moments)
            moments = None

            x_new += self.meas_z(z).unsqueeze(1)
            x_new = F.relu(x_new)
            # sum over neighbors to create new measure of size (batch_size,npoints,d_out)
            x_new = torch.mean(x_new.view(batch_size,
                                            npoints,
                                            N-1,
                                            self.d_out), 2).view(batch_size, npoints * self.d_out)

        z_new = self.vect_x(v_moments)
        v_moments = None
        z_new += self.vect_z(z).unsqueeze(1)
        z_new = F.relu(z_new)
        z_new = torch.mean(z_new, 1)

        return x_new, z_new


class EB_variable(EB):
    """ Pairwise interactions block.
    """
    def __init__(self, d_feat, d_Mfeat, d_out, N, npoints, nmoments, previous_nmoments, orders, position, with_label):
        super(EB_variable, self).__init__(d_feat, d_Mfeat, d_out, N, npoints, nmoments, previous_nmoments, orders, position, with_label)

    def forward(self, X, labels, z, info):
        x_new_list = []
        z_new_list = []

        list_npoints = []

        for i in range(len(X)):
            npoints = info[i][1].item()
            if len(X[i].size()) == 1:
                total_length_features = X[i, :info[i][0]].size(0)
                d_feat = int(total_length_features / npoints)
            else:
                d_feat = int(X[i].size(1) / npoints)

            len_dist = 0 #info[i][2].item()

            self.d_feat = d_feat

            x_new_, z_new_ = super(EB_variable, self).forward(
                                         X[i, :info[i][0]].view(npoints * d_feat).unsqueeze(0),
                                         z[i].unsqueeze(0),
                                         labels[i][:npoints].unsqueeze(0))

            x_new_list.append(x_new_.squeeze(0))
            z_new_list.append(z_new_.squeeze(0))
            list_npoints.append([self.d_out * npoints, npoints, len_dist])

        final_x = []

        min_npoints = min([a for _, a, _ in list_npoints])

        for i in range(len(X)):
            x_ = x_new_list[i].reshape(list_npoints[i][1], self.d_out)

            # print(x_new_list[i].size())
            x_ = x_[torch.LongTensor(min_npoints).random_(0, list_npoints[i][1]).long(), :].view(-1)
            # pad_x1 = torch.zeros(size=(int(np.max([v[0] * v[1] for v in list_npoints])), ))
            # pad_x1[:x_new_list[i].size(0)] = x_new_list[i]
            final_x.append(x_)

        return torch.stack(final_x), torch.stack(z_new_list), min_npoints
