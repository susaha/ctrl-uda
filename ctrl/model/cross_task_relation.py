from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


def get_bins(type, K):
    d = []
    if type == 'SID':
        a = 1
        b = 655.36
        for i in range(K):
            t_i = np.exp(np.log(a) + (np.log(b / a) * i) / K)
            d.append(np.ceil(t_i))
    elif type == 'SID2': # the bin process as n the CTRL cvpr21 paper
        a = 1
        b = 655.36
        for i in range(K):
            p1 = 1 - (2 * i + 1) / (2 * K)
            p2 = (2 * i + 1) / (2 * K)
            t_i = (a ** p1) * (b ** p2)
            d.append(t_i)
    d.sort()
    print('Depth bins in meters:')
    for i in range(K):
        print('{:.2f}'.format(d[i]), end=" ")
    print()
    return d


class DepthProb(nn.Module):
    def __init__(self, cfg):
        super(DepthProb, self).__init__()
        self.num_depth_bins = cfg.NUM_DEPTH_BINS
        self.sig = 1
        d = get_bins(cfg.DEPTH_SAMPLE_TYPE, self.num_depth_bins)
        depth_intval = torch.zeros(self.num_depth_bins, dtype=torch.float)
        for i in range(self.num_depth_bins):
            depth_intval[i] = d[i] * 100.0  # m --> cm
        print('Depth bins in cm:')
        for i in range(self.num_depth_bins):
            print('{:.2f}'.format(depth_intval[i]), end=" ")
        print()
        depth_intval = 65536.0 / (depth_intval + 1)  # converting the depth bins to inverse depth bins
        self.depint = depth_intval.to(cfg.GPU_ID)
        print('Inverse Depth bins in cm:')
        for i in range(self.num_depth_bins):
            print('{:.2f}'.format(self.depint[i]), end=" ")
        print()

    def forward(self, feat):
        c = []
        for i in range(self.num_depth_bins):
            c.append(-torch.square((feat - self.depint[i])) / self.sig)
        return F.softmax(torch.cat(c, 1), dim=1)


class Prob2Entropy(nn.Module):
    def __init__(self):
        super(Prob2Entropy, self).__init__()

    def forward(self, prob):
        n, c, h, w = prob.size()
        return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


class CTRL(nn.Module):
    def __init__(self, cfg):
        super(CTRL, self).__init__()
        print('ctrl/model/cross_task_relation.py --> class CTRL --> __init__()')
        self.get_depth_prob = DepthProb(cfg)
        self.prob_to_entropy = Prob2Entropy()

    def forward(self, semseg_pred, srh_pred, depth_pred):
        depth_prob = self.get_depth_prob(depth_pred)
        semseg_prob = F.softmax(semseg_pred, dim=1)
        srh_prob = F.softmax(srh_pred, dim=1)
        srh_prob = srh_prob.div(srh_prob.sum(dim=1, keepdim=True) + 1e-30) # not required
        semseg_entropy = self.prob_to_entropy(semseg_prob)
        srh_entropy = self.prob_to_entropy(srh_prob)
        depth_entropy = self.prob_to_entropy(depth_prob)
        FusedE = torch.cat((semseg_entropy, depth_entropy, srh_entropy), dim=1)
        return FusedE





