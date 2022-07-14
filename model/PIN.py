"""
Product Network in Network
"""

import torch
import torch.nn as nn


from deepctr_torch.models.basemodel import BaseModel 
from deepctr_torch.inputs import combined_dnn_input, SparseFeat, VarLenSparseFeat
from deepctr_torch.layers import DNN, concat_fun


class CatInnerProductLayer(nn.Module):
    """CatInnerProduct Layer used in PIN that compute the element-wise
    product between feature vectors and concat to feature vectors.
    """

    def __init__(self, device='cpu'):
        super(CatInnerProductLayer, self).__init__()
        self.to(device)

    def forward(self, inputs):

        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)

        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        
        p = torch.cat([embed_list[idx] for idx in row], dim=1)  # batch num_pairs k
        q = torch.cat([embed_list[idx] for idx in col], dim=1)

        inner_product = p * q
        
        output = torch.cat([p, q, inner_product], dim=2)
        return output


class PIN(BaseModel):
    """Instantiates the Product-based Neural Network architecture.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float . L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.
    """

    def __init__(self, 
    dnn_feature_columns, dnn_hidden_units=(128, 128), l2_reg_embedding=1e-5, l2_reg_dnn=0, 
    init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', 
    task='binary', device='cpu', gpus=None):

        super(PIN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)

        self.task = task

        num_inputs = self.compute_input_dim(dnn_feature_columns, include_dense=False, feature_group=True)
        self.product_out_dim = int(num_inputs * (num_inputs - 1) / 2)
        self.catinnerproduct = CatInnerProductLayer(device=device)

        sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat) or isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
        
        sparse_embedd_dim_list = [sp_f.embedding_dim for sp_f in sparse_feature_columns]
        embedd_dim = sparse_embedd_dim_list[0]
        
        self.inner_dnn_list = nn.ModuleList([
            nn.Linear(int(3 * embedd_dim), embedd_dim) # input dim:cat([v_i, v_j, vi \odot vj])
            for _ in range(self.product_out_dim)
        ])
        self.relu = nn.ReLU()

        self.dnn = DNN(
            (sparse_embedd_dim_list[0] * self.product_out_dim)  + self.compute_input_dim(dnn_feature_columns),
            dnn_hidden_units,
            activation=dnn_activation, 
            l2_reg=l2_reg_dnn, 
            dropout_rate=dnn_dropout, 
            use_bn=False,
            init_std=init_std, 
            device=device,
        )

        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        
        linear_signal = torch.flatten(concat_fun(sparse_embedding_list), start_dim=1)

        inner_product = self.catinnerproduct(sparse_embedding_list)
        
        # [product dim, batch size, embedding dim]
        inner_product = torch.stack([self.relu(self.inner_dnn_list[i](inner_product[:,i,:])) for i in range(self.product_out_dim)]) 
        # [product dim, batch size, embedding dim] -> [batch size, product dim, embedding dim]
        inner_product = inner_product.permute(1,0,2)
        # [batch size, product dim, embedding dim] -> [batch size, product dim * embedding dim]
        inner_product = torch.flatten(inner_product, start_dim=1)
        
        product_layer = torch.cat([linear_signal, inner_product], dim=1)

        dnn_input = combined_dnn_input([product_layer], dense_value_list)
        
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = dnn_logit

        y_pred = self.out(logit)

        return y_pred