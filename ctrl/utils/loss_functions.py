import torch.nn as nn
from torch.autograd import Variable
import torch


class CrossEntropy2D(nn.Module):
    def __init__(self):
        super(CrossEntropy2D, self).__init__()
        print('ctrl/utils/loss_functions.py --> class CrossEntropy2D --> __init__()')
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, predict, target):
        assert not target.requires_grad
        target = target.long()
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
        assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
        assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != 255)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = self.criterion(predict, target)
        return loss


class BerHuLossDepth(nn.Module):
    def __init__(self):
        super(BerHuLossDepth, self).__init__()
        print('ctrl/utils/loss_functions.py --> class BerHuLossDepth --> __init__()')

    def forward(self, pred, label, is_vector=None):
        if not is_vector:
            n, c, h, w = pred.size()
            assert c == 1
            pred = pred.squeeze()
            label = label.squeeze()
        # label = label.squeeze().to(self.device)
        adiff = torch.abs(pred - label)
        batch_max = 0.2 * torch.max(adiff).item()
        t1_mask = adiff.le(batch_max).float()
        t2_mask = adiff.gt(batch_max).float()
        t1 = adiff * t1_mask
        t2 = (adiff * adiff + batch_max * batch_max) / (2 * batch_max)
        t2 = t2 * t2_mask
        return (torch.sum(t1) + torch.sum(t2)) / torch.numel(pred.data)


class BCELossSS(nn.Module):
    def __init__(self):
        super(BCELossSS, self).__init__()
        print('ctrl/utils/loss_functions.py --> class BCELossSS --> __init__()')

    def forward(self, y_pred, y_label):
        y_truth_tensor = torch.FloatTensor(y_pred.size())
        y_truth_tensor.fill_(y_label)
        y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
        return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)



