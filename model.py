from torch import nn


class Model(nn.Module):
    def __init__(self, text, num_class):
        super().__init__()

        vocab_size = text.vocab.vectors.size(0)
        embed_dim = text.vocab.vectors.size(1)
        hidden_dim = 512

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=4, dropout=0.2, bidirectional=True)

        self.embedding.weight.data.copy_(text.vocab.vectors)
        self.embedding.weight.requires_grad = False

        self.linear = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden_dim * 2, num_class))

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.encoder(embed)

        out = self.linear(out[-1])
        return out
