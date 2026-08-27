# Model with feature visualization

from torch import nn
from torchvision import models
from .frequency_feature import LightweightFrequencyFeature


class Model(nn.Module):

    def __init__(
        self,
        num_classes,
        latent_dim=2048,
        lstm_layers=1,
        hidden_dim=2048,
        bidirectional=False,
        use_frequency_feature=True
    ):
        super(Model, self).__init__()

        self.use_frequency_feature = use_frequency_feature
        if self.use_frequency_feature:
            self.freq_feature = LightweightFrequencyFeature(high_freq_threshold=0.5)

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

        freq_score = None
        if self.use_frequency_feature:
            freq_score = self.freq_feature(x)

        x_reshaped = x.view(batch_size * seq_length, c, h, w)

        fmap = self.model(x_reshaped)

        x_pooled = self.avgpool(fmap)

        x_pooled = x_pooled.view(batch_size, seq_length, 2048)

        x_lstm, _ = self.lstm(x_pooled, None)
        
        logits = self.dp(self.linear1(x_lstm[:, -1, :]))

        if self.use_frequency_feature:
            return fmap, logits, freq_score
        return fmap, logits
