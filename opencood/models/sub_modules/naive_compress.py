import torch
import torch.nn as nn

from opencood.models.sub_modules.CoE2E_utils import LayerNorm

class NaiveCompressor(nn.Module):
    """
    A very naive compression that only compress on the channel.
    """
    def __init__(self, input_dim, compress_raito):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim//compress_raito, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim//compress_raito, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim//compress_raito, input_dim, kernel_size=3,
                      stride=1, padding=1),
            LayerNorm(normalized_shape=input_dim, eps=1e-6, data_format='channels_first'),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
            LayerNorm(normalized_shape=input_dim, eps=1e-6, data_format='channels_first'),
            nn.ReLU()
        )

    def compress(self, x):
        x = self.encoder(x)
        return x

    def decompress(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x



class NaiveCompressorUMC(nn.Module):
    """
    A very naive compression that only compress on the channel.
    """
    def __init__(self, input_dim, compress_raito):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim//compress_raito, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim//compress_raito, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim//compress_raito, input_dim//2, kernel_size=3,
                      stride=1, padding=1),
            LayerNorm(normalized_shape=input_dim//2, eps=1e-6, data_format='channels_first'),
            nn.ReLU(),
            nn.Conv2d(input_dim//2, input_dim//2, kernel_size=3, stride=1, padding=1),
            LayerNorm(normalized_shape=input_dim//2, eps=1e-6, data_format='channels_first'),
            nn.ReLU()
        )

    def compress(self, x):
        x = self.encoder(x)
        return x

    def decompress(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x