import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, input):
        return self.main(input)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, input):
        return self.main(input)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.encoder = nn.ModuleList([
            DownSamplingLayer(1, 24),     # 16384 => 8192
            DownSamplingLayer(24, 48),    # 4096
            DownSamplingLayer(48, 72),    # 2048

            DownSamplingLayer(72, 96),    # 1024
            DownSamplingLayer(96, 120),   # 512
            DownSamplingLayer(120, 144),  # 256
            
            DownSamplingLayer(144, 168),  # 128
            DownSamplingLayer(168, 192),  # 64
            DownSamplingLayer(192, 216),  # 32
            
            DownSamplingLayer(216, 240),  # 16
            DownSamplingLayer(240, 264),  # 8
            DownSamplingLayer(264, 288),  # 4
        ])

        self.middle = nn.Sequential(
            nn.Conv1d(288, 288, 15, stride=1, padding=7),
            nn.BatchNorm1d(288),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.decoder = nn.ModuleList([
            UpSamplingLayer(288 + 288, 288),
            UpSamplingLayer(264 + 288, 264), # 同水平层的降采样后维度为 264
            UpSamplingLayer(240 + 264, 240),

            UpSamplingLayer(216 + 240, 216),
            UpSamplingLayer(192 + 216, 192),
            UpSamplingLayer(168 + 192, 168),

            UpSamplingLayer(144 + 168, 144),
            UpSamplingLayer(120 + 144, 120),
            UpSamplingLayer(96 + 120, 96),

            UpSamplingLayer(72 + 96, 72),
            UpSamplingLayer(48 + 72, 48),
            UpSamplingLayer(24 + 48, 24),
        ])

        self.out = nn.Sequential(
            nn.Conv1d(1 + 24, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input):
        tmp = []
        o = input

        # Up Sampling
        for i in range(12):
            o = self.encoder[i](o)
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        o = self.middle(o)

        # Down Sampling
        for i in range(12):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            o = torch.cat([o, tmp[12 - i - 1]], dim=1)
            o = self.decoder[i](o)
        
        o = torch.cat([o, input], dim=1)
        o = self.out(o)
        return o