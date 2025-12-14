import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(
        self,
        num_input_channels: int,
        base_channel_size: int,
        latent_dim: int,
        act_fn=nn.GELU
    ):
        super().__init__()
        c_hid = base_channel_size

        self.conv_net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, 3, stride=2, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, 3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, 3, stride=2, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, 3, padding=1),
            act_fn(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(2 * c_hid * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(2 * c_hid * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.conv_net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
