from torch import nn


class Model(nn.Module):
    def __init__(self, text, hidden_dim=512, num_layers=2, num_class=5):
        super().__init__()

        vocab_size = text.vocab.vectors.size(0)
        embed_dim = text.vocab.vectors.size(1)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, dropout=0.5, bidirectional=True)

        self.embedding.weight.data.copy_(text.vocab.vectors)
        self.embedding.weight.requires_grad = False

        self.linear = nn.Sequential(nn.Dropout(0.5), nn.Linear(hidden_dim * 2, num_class))

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.encoder(embed)

        out = self.linear(out[-1])
        return out
