import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
from loss import main_loss
import math


class PositionalEncoding(nn.Module):
    def __init__(self, dim_in=512, dim_emb=256, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, dim_emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_emb, 2).float() * (-math.log(10000.0) / dim_emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.fc = nn.Linear(dim_in + 2 * dim_emb, dim_in)

    def forward(self, x, pos):
        '''
        :param x: (B, W, H, D)
        :param pos: (B, 2) -> (x, y)
        :return:
        '''
        B, W, H, D = x.size()
        pe_table = self.pe.repeat(B, 1, 1)
        pos_w = pe_table[torch.arange(B), pos[:, 0], :]  # (B, D)
        pos_h = pe_table[torch.arange(B), pos[:, 1], :]
        pos_w = pos_w.view(B, 1, 1, -1).repeat(1, W, H, 1)
        pos_h = pos_h.view(B, 1, 1, -1).repeat(1, W, H, 1)
        x = torch.cat([x, pos_w, pos_h], dim=-1)  # (B, W, H, 3*D)
        x = self.fc(x)
        return x


def conv2d_bn_relu(dim_in, dim_out, k, padding):
    layer = nn.Sequential(
        nn.Conv2d(dim_in, dim_out, k, padding=padding),
        nn.BatchNorm2d(dim_out),
        nn.ReLU(inplace=True)
    )
    return layer


class DoubleConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, k=3, stride=1, padding=1):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, k, stride, padding),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, k, stride, padding),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv2d(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256], kernel=3, device='cpu'):
        super().__init__()
        self.device = device
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pe = PositionalEncoding(dim_in=features[0], dim_emb=features[0], max_len=36)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv2d(in_channels, feature, kernel, padding=kernel//2))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv2d(feature * 2, feature, kernel, padding=kernel//2))
        self.bottleneck = DoubleConv2d(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # image_batch: B, D, H, W
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        pred = self.final_conv(x)  # B, 1, H, W
        return pred.squeeze(1)

    
class UNet_deep(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], kernel=3, device='cpu'):
        super().__init__()
        self.device = device
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv2d(in_channels, feature, kernel, padding=kernel//2))
            in_channels = feature
        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature, feature//2, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv2d(feature * 2, feature, kernel, padding=kernel//2))
        self.bottleneck = DoubleConv2d(features[-1], features[-1])
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0], kernel_size=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], out_channels, kernel_size=1)
        )

    def forward(self, x):
        # image_batch: B, D, H, W
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        pred = self.final_conv(x)  # B, 1, H, W
        return pred.squeeze(1)
    
    
class seg_model(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.unet = UNet(in_channels=1, out_channels=1, features=config.model.feature,
                         kernel=config.model.kernel, device=device)
        self.loss = main_loss()

    def forward(self, image_batch, label_batch, image_pos=None, image_posr=None):
        input_tensor = image_batch.permute(0, 3, 1, 2)  # B, H, W, D -> B, D, H, W
        pred = self.unet(input_tensor)
        loss = self.loss(pred, label_batch)
        rgl_loss = sum(p.pow(2.0).sum() for p in self.unet.parameters())
        loss['rgl_loss'] = rgl_loss
        return pred, loss


if __name__ == "__main__":
    B, N, D = (2, 48, 512)
    # x = torch.randn(size=(2, 512, 64, 64))  # B, N, D
    x = torch.randn(size=(2, 64, 64, 512))  # B, N, D
    pos = torch.tensor([[0,0], [1,1]])
    model = PositionalEncoding()
    out = model(x, pos)
    print(out.shape)
    pass
