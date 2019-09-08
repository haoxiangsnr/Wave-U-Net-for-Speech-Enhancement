import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNet(nn.Module):
    def __init__(self, nefilters=24):
        super(UNet, self).__init__()
        print('random unet')
        nlayers = 12
        self.num_layers = nlayers
        self.nefilters = nefilters
        filter_size = 5
        merge_filter_size = 5
        self.context = True
        self.encoder0 = nn.ModuleList()
        self.encoder1 = nn.ModuleList()
        self.encoder2 = nn.ModuleList()
        self.decoder0 = nn.ModuleList()
        self.decoder1 = nn.ModuleList()
        self.decoder2 = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dbatch = nn.ModuleList()

        echannelin = [1] + [(i + 1) * nefilters for i in range(nlayers - 1)]
        echannelout = [(i + 1) * nefilters for i in range(nlayers)]
        dchannelout = echannelout[::-1]
        dchannelin = [dchannelout[0] * 2] + [(i) * nefilters + (i - 1) * nefilters for i in range(nlayers, 1, -1)]

        for i in range(self.num_layers):
            self.encoder0.append(
                nn.Conv1d(echannelin[i], echannelout[i], filter_size, dilation=1, padding=filter_size // 2 * 1))
            self.encoder1.append(
                nn.Conv1d(echannelin[i], echannelout[i], filter_size, dilation=2, padding=filter_size // 2 * 2))
            self.encoder2.append(
                nn.Conv1d(echannelin[i], echannelout[i], filter_size, dilation=3, padding=filter_size // 2 * 3))

            self.decoder0.append(nn.Conv1d(dchannelin[i], dchannelout[i], merge_filter_size, dilation=1,
                                           padding=merge_filter_size // 2 * 1))
            self.decoder1.append(nn.Conv1d(dchannelin[i], dchannelout[i], merge_filter_size, dilation=2,
                                           padding=merge_filter_size // 2 * 2))
            self.decoder2.append(nn.Conv1d(dchannelin[i], dchannelout[i], merge_filter_size, dilation=3,
                                           padding=merge_filter_size // 2 * 3))

            self.ebatch.append(nn.BatchNorm1d(echannelout[i]))
            self.dbatch.append(nn.BatchNorm1d(dchannelout[i]))

        self.encoder = [self.encoder0, self.encoder1, self.encoder2]
        # self.encoder.append(self.encoder0)
        # self.encoder.append(self.encoder1)
        # self.encoder.append(self.encoder2)
        self.decoder = [self.decoder0, self.decoder1, self.decoder2]
        # self.decoder.append(self.decoder0)
        # self.decoder.append(self.decoder1)
        # self.decoder.append(self.decoder2)

        self.middle = nn.Sequential(
            nn.Conv1d(echannelout[-1], echannelout[-1], filter_size, padding=filter_size // 2),
            nn.BatchNorm1d(echannelout[-1]),
            nn.LeakyReLU(0.1)
        )
        self.out = nn.Sequential(
            nn.Conv1d(nefilters + 1, 1, 1),
            nn.Tanh()
        )


    def forward(self, x, randint=None):
        if not randint:
            randint = np.random.randint(0, 3)

        input = x
        encoder = list()
        for i in range(self.num_layers):
            # print(randint)
            x = self.encoder[int(randint[i])][i](x)
            # x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x, 0.1)
            encoder.append(x)
            x = x[:, :, ::2]

        x = self.middle(x)

        for i in range(self.num_layers):
            x = F.upsample(x, scale_factor=2, mode='linear')
            x = torch.cat([x, encoder[self.num_layers - i - 1]], dim=1)
            x = self.decoder[int(randint[i + self.num_layers])][i](x)
            x = self.dbatch[i](x)
            x = F.leaky_relu(x, 0.1)
        x = torch.cat([x, input], dim=1)

        x = self.out(x)
        return x
