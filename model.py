import math

import torch
import torch.nn.functional as F
from capsule_layer import CapsuleLinear
from torch import nn
from torch.nn.parameter import Parameter


class CompositionalEmbedding(nn.Module):
    r"""A simple compositional codeword and codebook that store embeddings.

     Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): size of each embedding vector
        num_codebook (int): size of the codebook of embeddings
        num_codeword (int, optional): size of the codeword of embeddings

     Shape:
         - Input: (LongTensor): (N, W), W = number of indices to extract per mini-batch
         - Output: (Tensor): (N, W, embedding_dim)

     Attributes:
         - code (Tensor): the learnable weights of the module of shape
              (num_embeddings, num_codebook, num_codeword)
         - codebook (Tensor): the learnable weights of the module of shape
              (num_codebook, num_codeword, embedding_dim)

     Examples::
         >>> from torch.autograd import Variable
         >>> m = CompositionalEmbedding(20000, 64, 16, 32)
         >>> input = Variable(torch.randperm(128).view(16, -1))
         >>> output = m(input)
         >>> print(output.size())
         torch.Size([16, 8, 64])
     """

    def __init__(self, num_embeddings, embedding_dim, num_codebook, num_codeword=None):
        super(CompositionalEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if num_codeword is None:
            num_codeword = math.ceil(math.pow(num_embeddings, 1 / num_codebook))
        self.code = Parameter(torch.Tensor(num_embeddings, num_codebook, num_codeword))
        self.codebook = Parameter(torch.Tensor(num_codebook, num_codeword, embedding_dim))

        nn.init.normal(self.code)
        nn.init.normal(self.codebook)

    def forward(self, input):
        batch_size = input.size(0)
        index = input.view(-1)
        code = self.code.index_select(dim=0, index=index)
        # reweight, do softmax, make sure the sum of weight about each book to 1
        code = F.softmax(code, dim=-2)
        out = (code[:, :, None, :] @ self.codebook[None, :, :, :]).squeeze(dim=-2).sum(dim=1)
        out = out.view(batch_size, -1, self.embedding_dim)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.num_embeddings) + ', ' + str(self.embedding_dim) + ')'


class Model(nn.Module):
    def __init__(self, vocab_size, num_class, num_iterations):
        super().__init__()

        self.embedding = CompositionalEmbedding(num_embeddings=vocab_size, embedding_dim=64, num_codebook=8)
        self.features = nn.LSTM(64, 128, num_layers=2, dropout=0.2, batch_first=True, bidirectional=True)

        self.classifier = CapsuleLinear(out_capsules=num_class, in_length=16, out_length=8,
                                        num_iterations=num_iterations)

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.features(embed)

        out = out[:, -1, :].contiguous().view(out.size(0), -1, 16)
        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes
