import torch
import torch.nn as nn
from torchvision.models import convnext_base
from Config import HyperParams

class SFD(nn.Module):
    def __init__(self, in_channel, k):
        super(SFD, self).__init__()
        self.k = k
        self.stripconv = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, 1, 0),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv = nn.Sequential(
            nn.Conv2d(2, 2, 1, 1, 0),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU())

    def forward(self, fm):
        b, c, w, h = fm.shape
        fms_w = torch.split(fm, int(w //self.k), dim=2)
        fms_h = torch.split(fm, int(h //self.k), dim=3)

        fms_conv_w = map(self.stripconv, fms_w)
        fms_conv_h = map(self.stripconv, fms_h)

        fms_pool_w = list(map(self.avgpool, fms_conv_w))
        fms_pool_h = list(map(self.avgpool, fms_conv_h))

        fms_pool_w = torch.cat(fms_pool_w, dim=2)
        fms_pool_h = torch.cat(fms_pool_h, dim=3)

        fms_pool_w =  fms_pool_w.permute(0, 1, 3, 2)

        w_h = torch.cat([fms_pool_w, fms_pool_h], dim=1)
        w_h = self.conv(w_h)
        w_h_c = torch.chunk(w_h, 2,dim=1)

        fms_pool_w =  w_h_c[0].permute(0, 1, 3, 2)
        fms_pool_h =  w_h_c[1]

        fms_softmax_w = torch.softmax(fms_pool_w, dim=2)
        fms_softmax_h = torch.softmax(fms_pool_h, dim=3)

        fms_softmax_boost_w = torch.repeat_interleave( fms_softmax_w, int(w //self.k), dim=2)
        fms_softmax_boost_h = torch.repeat_interleave( fms_softmax_h, int(h //self.k), dim=3)

        alpha = HyperParams['alpha']
        fms_boost = fm + alpha*(fm * fms_softmax_boost_w * fms_softmax_boost_h)

        beta = HyperParams['beta']
        fms_max_w = torch.max(fms_softmax_w, dim=2, keepdim=True)[0]
        fms_max_h = torch.max(fms_softmax_h, dim=3, keepdim=True)[0]

        fms_softmax_suppress_w = torch.clamp((fms_softmax_w < fms_max_w).float(), min=beta)
        fms_softmax_suppress_h= torch.clamp((fms_softmax_h < fms_max_h).float(), min=beta)

        fms_softmax_suppress_w1 = torch.repeat_interleave(fms_softmax_suppress_w, w // self.k, dim=2)
        fms_softmax_suppress_h1 = torch.repeat_interleave(fms_softmax_suppress_h, h // self.k, dim=3)
        fms_suppress = fm * fms_softmax_suppress_w1*fms_softmax_suppress_h1

        return fms_boost, fms_suppress

class ConvNext(nn.Module):
    def __init__(self):
        super(ConvNext, self).__init__()
        self.model = list(convnext_base(pretrained=True).features.children())
        self.layer0_2 = nn.Sequential(*self.model[:4])
        self.layer3 = nn.Sequential(*self.model[4:6])
        self.layer4 = nn.Sequential(*self.model[6:])
        self.E_S1 = SFD(in_channel=256, k=7)
        self.E_S2 = SFD(in_channel=512, k=7)
        self.E_S3 = SFD(in_channel=1024, k=7)
    def forward(self, x):
        fm2 = self.layer0_2(x)
        fm2_enhance, fm2_suppress = self.E_S1(fm2)
        fm3 = self.layer3(fm2_suppress)
        fm3_enhance, fm3_suppess = self.E_S2(fm3)
        fm4 = self.layer4(fm3_suppess)
        fm4_enhance, _ = self.E_S3(fm4)
        return fm2_enhance, fm3_enhance, fm4_enhance
    def get_params(self):
        new_layers = list(self.strip1.parameters()) + \
                     list(self.strip2.parameters()) + \
                     list(self.strip3.parameters())
        new_layers_id = list(map(id, new_layers))
        old_layers = filter(lambda p: id(p) not in new_layers_id, self.parameters())
        return new_layers, old_layers



