import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNet(nn.Module):
    def __init__(self,nefilters=24):
        super(UNet, self).__init__()
        print('pyramid unet')
        nlayers = 12
        self.num_layers = nlayers
        self.nefilters = nefilters
        filter_size = 15
        merge_filter_size = 5
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dbatch = nn.ModuleList()
        echannelin = [1] + [(i + 1) * nefilters for i in range(nlayers-1)]
        echannelout = [(i + 1) * nefilters for i in range(nlayers)]
        dchannelout = echannelout[::-1]
        dchannelin = [dchannelout[0]*2]+[(i) * nefilters + (i - 1) * nefilters for i in range(nlayers,1,-1)]
        for i in range(self.num_layers):
            self.encoder.append(nn.Conv1d(echannelin[i],echannelout[i],filter_size,padding=filter_size//2))
            self.decoder.append(nn.Conv1d(dchannelin[i],dchannelout[i],merge_filter_size,padding=merge_filter_size//2))
            self.ebatch.append(nn.BatchNorm1d(echannelout[i]))
            self.dbatch.append(nn.BatchNorm1d(dchannelout[i]))
        rates = [1, 2, 3, 4]
        self.aspp1 = ASPP(echannelout[-1], echannelout[-1]//4, rate=rates[0])
        self.aspp2 = ASPP(echannelout[-1], echannelout[-1]//4, rate=rates[1])
        self.aspp3 = ASPP(echannelout[-1], echannelout[-1]//4, rate=rates[2])
        self.aspp4 = ASPP(echannelout[-1], echannelout[-1]//4, rate=rates[3])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                             nn.Conv1d(echannelout[-1], echannelout[-1]//4, 1, bias=False),
                                             nn.BatchNorm1d(echannelout[-1]//4),
                                             nn.LeakyReLU(0.1))
        self.middle = nn.Sequential(
            nn.Conv1d(echannelout[-1]//4*5, echannelout[-1], 1, bias=False),
            nn.BatchNorm1d(echannelout[-1]),
            nn.LeakyReLU(0.1)
        )
        self.out = nn.Sequential(
            nn.Conv1d(nefilters + 1, 1, 1),
            nn.Tanh()
        )
    def forward(self,x):
        encoder = list()
        input = x
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x,0.1)
            encoder.append(x)
            x = x[:,:,::2]

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='linear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.middle(x)

        for i in range(self.num_layers):
            # x = F.upsample(x,scale_factor=2,mode='linear')
            x = torch.cat([x,x], dim=-1)
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)
            x = self.decoder[i](x)
            x = self.dbatch[i](x)
            x = F.leaky_relu(x,0.1)
        x = torch.cat([x,input],dim=1)

        x = self.out(x)
        return x

class ASPP(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.convbnre = nn.Sequential(
            nn.Conv1d(inplanes, planes, kernel_size=kernel_size,
                      stride=1, padding=padding, dilation=rate, bias=False),
            nn.BatchNorm1d(planes),
            nn.LeakyReLU(0.1)
        )
        self.__init_weight()

    def forward(self, x):
        return self.convbnre(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.dataset.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()