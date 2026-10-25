"""PyTorch models used by the data quality checks."""
import torch
import torch.nn as nn


class NewsClassifier(nn.Module):
    """Embedding -> BiGRU -> (max + mean pooling) -> BatchNorm/Dropout MLP -> 4 classes.

    Trained from scratch on AG News. Inside the recommender it is used to find rows whose
    label disagrees confidently with the text (likely label noise).
    """

    def __init__(self, vocab_size, embed_dim=128, hidden=128, n_classes=4, dropout=0.4):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, embed_dim=embed_dim, hidden=hidden,
                           n_classes=n_classes, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(dropout / 2)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.BatchNorm1d(4 * hidden),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def pooled(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        h, _ = self.gru(self.embed_drop(self.embedding(x)))
        # max pooling ignores padding by pushing padded steps to a large negative value
        h_max = (h - (1 - mask) * 1e4).max(dim=1).values
        h_mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return torch.cat([h_max, h_mean], dim=1)

    def forward(self, x):
        return self.head(self.pooled(x))


class CharAutoencoder(nn.Module):
    """Denoising autoencoder over hashed character n-gram vectors.

    It learns what normal news text looks like at the character level. Rows it cannot
    reconstruct well (high error) are garbled, heavily misspelled or off-domain.
    """

    def __init__(self, n_features=4096, hidden=512, bottleneck=64, dropout=0.2, input_noise=0.1):
        super().__init__()
        self.config = dict(n_features=n_features, hidden=hidden, bottleneck=bottleneck,
                           dropout=dropout, input_noise=input_noise)
        self.input_noise = nn.Dropout(input_noise)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(self.input_noise(x)))


def reconstruction_error(model, x):
    """Per-row mean squared reconstruction error (model must be in eval mode)."""
    with torch.no_grad():
        return ((model(x) - x) ** 2).mean(dim=1)
