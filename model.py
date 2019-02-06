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
         >>> m = CompositionalEmbedding(20000, 64, 16, 32)
         >>> a = torch.randperm(128).view(16, -1)
         >>> output = m(a)
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

        nn.init.normal_(self.code)
        nn.init.normal_(self.codebook)

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
    def __init__(self, vocab_size, num_class, routing_type, num_iterations):
        super().__init__()

        self.embedding_size = 64
        self.hidden_size = 128

        self.embedding = CompositionalEmbedding(num_embeddings=vocab_size, embedding_dim=self.embedding_size,
                                                num_codebook=8)
        self.features = nn.GRU(self.embedding_size, self.hidden_size, num_layers=2, dropout=0.5, batch_first=True,
                               bidirectional=True)
        if routing_type == 'k_means':
            self.classifier = CapsuleLinear(out_capsules=num_class, in_length=8, out_length=16, in_capsules=16,
                                            share_weight=False, num_iterations=num_iterations, similarity='cosine')
        else:
            self.classifier = CapsuleLinear(out_capsules=num_class, in_length=8, out_length=16, in_capsules=16,
                                            share_weight=False, routing_type='dynamic', num_iterations=num_iterations)

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.features(embed)

        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        out = out.mean(dim=1).contiguous().view(out.size(0), -1, 8)
        out = self.classifier(out)
        classes = out.norm(dim=-1)
        return classes
