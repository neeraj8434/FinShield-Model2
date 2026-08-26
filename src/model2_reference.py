# Model with feature visualization

from torch import nn
from torchvision import models


class Model(nn.Module):

    def __init__(
        self,
        num_classes,
        latent_dim=2048,
        lstm_layers=1,
        hidden_dim=2048,
        bidirectional=False
    ):
        super(Model, self).__init__()

        model = models.resnext50_32x4d(pretrained=True)

        self.model = nn.Sequential(*list(model.children())[:-2])

        self.lstm = nn.LSTM(
            latent_dim,
            hidden_dim,
            lstm_layers,
            bidirectional
        )

        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape

        x = x.view(batch_size * seq_length, c, h, w)

        fmap = self.model(x)

        x = self.avgpool(fmap)

        x = x.view(batch_size, seq_length, 2048)

        x_lstm, _ = self.lstm(x, None)

        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))
