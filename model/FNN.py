"""
FM-supported Neural Network
"""

import torch
import torch.nn as nn


from deepctr_torch.models.basemodel import BaseModel 
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN



class FNN(BaseModel):
    """
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.
    """

    def __init__(self,
                 dnn_feature_columns,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(FNN, self).__init__([], dnn_feature_columns, l2_reg_linear=l2_reg_linear,
        l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
        device=device, gpus=gpus)

        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                        activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                        init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)

        return y_pred