

from deepctr_torch import models
from model import PIN, FNN



def BuildModel(model_name, features_columns, device):
    if model_name == "deepfm":
        model = models.DeepFM(
            linear_feature_columns=features_columns,
            dnn_feature_columns=features_columns,
            task='regression',
            device=device,
        )
    elif model_name == "ipnn":
        model = models.PNN(
            dnn_feature_columns=features_columns,
            task='regression',
            use_inner=True,
            use_outter=False,
            device=device,
        )
    elif model_name == "opnn":
        model = models.PNN(
            dnn_feature_columns=features_columns,
            task='regression',
            use_inner=False,
            use_outter=True,
            device=device,
        )
    elif model_name == "pin":
        model = PIN(
            dnn_feature_columns=features_columns,
            task='regression',
            device=device,
        )
    elif model_name == "ccpm":
        model = models.CCPM(
            linear_feature_columns=features_columns,
            dnn_feature_columns=features_columns,
            task='regression',
            device=device,
        )
    elif model_name == "wd":
        model = models.WDL(
            linear_feature_columns=features_columns,
            dnn_feature_columns=features_columns,
            task='regression',
            device=device,
        )
    elif model_name == "afm":
        model = models.AFM(
            linear_feature_columns=features_columns,
            dnn_feature_columns=features_columns,
            task='regression',
            device=device,
        )
    elif model_name == "nfm":
        model = models.NFM(
            linear_feature_columns=features_columns,
            dnn_feature_columns=features_columns,
            task='regression',
            device=device,
        )
    elif model_name == "xdeepfm":
        model = models.xDeepFM(
            linear_feature_columns=features_columns,
            dnn_feature_columns=features_columns,
            task='regression',
            device=device,
        )
    elif model_name == "fnn":
        model = FNN(
            dnn_feature_columns=features_columns,
            task='regression',
            device=device,
        )
    elif model_name == "dcn":
        model = models.DCN(
            linear_feature_columns=features_columns,
            dnn_feature_columns=features_columns,
            task='regression',
            device=device,
        )
    return model