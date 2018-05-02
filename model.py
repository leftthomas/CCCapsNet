from torch import nn


class Model(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300, recurrent_dropout=0.1):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(784, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1, dropout=recurrent_dropout)
        self.linear_layers = nn.Linear(hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, 6)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        feature = self.linear_layers(feature)
        preds = self.predictor(feature)
        return preds
