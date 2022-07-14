
# Resource: https://www.kaggle.com/code/gennadylaptev/factorization-machine-implemented-in-pytorch/notebook

import numpy as np
import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, field_dims, embed_dim, **kwargs):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(field_dims, embed_dim), requires_grad=True)
        self.lin = nn.Linear(field_dims, 1)

        
    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        
        return out.squeeze(1)



# Resource: https://github.com/rixwew/pytorch-fm

# import numpy as np
# import torch
# import torch.nn as nn


# class FeaturesLinear(nn.Module):

#     def __init__(self, field_dims, output_dim=1):
#         super().__init__()
#         self.fc = nn.Embedding(sum(field_dims), output_dim)
#         self.bias = nn.Parameter(torch.zeros((output_dim,)))
#         self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

#     def forward(self, x):
#         """
#         :param x: Long tensor of size ``(batch_size, num_fields)``
#         """
#         # x = x + x.new_tensor(self.offsets).unsqueeze(0)
#         return torch.sum(self.fc(x), dim=1) + self.bias


# class FeaturesEmbedding(nn.Module):

#     def __init__(self, field_dims, embed_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(sum(field_dims), embed_dim)
#         self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
#         nn.init.xavier_uniform_(self.embedding.weight.data)

#     def forward(self, x):
#         """
#         :param x: Long tensor of size ``(batch_size, num_fields)``
#         """
#         # x = x + x.new_tensor(self.offsets).unsqueeze(0)
#         return self.embedding(x)


# class FactorizationMachine(nn.Module):

#     def __init__(self, reduce_sum=True):
#         super().__init__()
#         self.reduce_sum = reduce_sum

#     def forward(self, x):
#         """
#         :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
#         """
#         square_of_sum = torch.sum(x, dim=1) ** 2
#         sum_of_square = torch.sum(x ** 2, dim=1)
#         ix = square_of_sum - sum_of_square
#         if self.reduce_sum:
#             ix = torch.sum(ix, dim=1, keepdim=True)
#         return 0.5 * ix

# class Model(nn.Module):
#     """
#     A pytorch implementation of Factorization Machine.
#     Reference:
#         S Rendle, Factorization Machines, 2010.
#     """

#     def __init__(self, field_dims, embed_dim, **kwargs):
#         super().__init__()
#         self.embedding = FeaturesEmbedding(field_dims, embed_dim)
#         self.linear = FeaturesLinear(field_dims)
#         self.fm = FactorizationMachine(reduce_sum=True)

#     def forward(self, x):
#         """
#         :param x: Long tensor of size ``(batch_size, num_fields)``
#         """
#         x = self.linear(x) + self.fm(self.embedding(x))
#         # return torch.sigmoid(x.squeeze(1))
#         return torch.sigmoid(x.squeeze(1)) * 5 # since score start from 1 to 5