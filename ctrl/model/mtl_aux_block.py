import torch.nn as nn
from ctrl.model.resnet_backbone import ResnetBackbone, Bottleneck
from ctrl.model.decoder import DecoderAuxBlock
from ctrl.model.decoder_final_classifer import DecFinalClassifier


class MTLAuxBlock(nn.Module):
    def __init__(self, n_classes, disbale_depth=False, disble_srh=False):
        super(MTLAuxBlock, self).__init__()
        print('ctrl/model/mtl_aux_block.py --> class MTLAuxBlock -->  def __init__()')
        self.backbone = ResnetBackbone(Bottleneck, [3, 4, 23, 3])
        self.decoder = DecoderAuxBlock(inpdim=128, outdim=2048)
        self.head_seg = DecFinalClassifier(n_classes, 2048)
        self.disbale_depth = disbale_depth
        self.disble_srh = disble_srh
        if not self.disbale_depth:
            self.head_dep = DecFinalClassifier(1, inpdim=128)
        if not self.disble_srh:
            self.head_srh = DecFinalClassifier(n_classes, inpdim=2048)

    def forward(self, x):
        ph = None
        dep_pred = None
        srh_pred = None
        x4 = self.backbone(x)
        x4_dec3, x4_dec4 = self.decoder(x4)
        if not self.disbale_depth:
            dep_pred = self.head_dep(x4_dec3)
        x4 = x4 * x4_dec4
        seg_pred = self.head_seg(x4)
        if not self.disble_srh:
            srh_pred = self.head_srh(x4 * dep_pred)
        return ph, seg_pred, dep_pred, srh_pred