from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(9449, 100)
        self.encoder = nn.LSTM(100, 500, num_layers=1, dropout=0.1)
        self.linear_layers = nn.Linear(500, 500)
        self.predictor = nn.Linear(500, 6)

    def forward(self, seq):
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        feature = self.linear_layers(feature)
        preds = self.predictor(feature)
        return preds
