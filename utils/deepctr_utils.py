
import numpy as np
import torch
from torch.utils.data import DataLoader
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat
from utils import utils, metrics

def GetFeatColumns(feat_type_dict):
    features_columns = []
    
    for k, v in feat_type_dict.items():
        
        if v["f_type"] == "Sparse":
            # sparse variable's "f_len" is equal to 1
            features_columns.append(SparseFeat(k + "_0", vocabulary_size=v["vocabulary_size"], embedding_dim=4))
            
        elif v["f_type"] == "VarLenSparse":
            # sparse variable's "f_len" is equal to 1
            features_columns.append(
                VarLenSparseFeat(
                    SparseFeat(k + "_0", vocabulary_size=v["vocabulary_size"], embedding_dim=4),
                    maxlen=v["max_len"], combiner="mean",
                )
            )
        else:
            for i in range(v["f_len"]):features_columns.append(DenseFeat(k + "_" + str(i), 1))

    return features_columns

def Predict(Id_info_dict, data_pair_dict, features_dict_dict, features_type_dict, model):
    
    all_pair_without_train = utils.GetAllPairWithoutTrainSet(
        user_id=Id_info_dict["user_id"],
        item_id_range=[Id_info_dict["item_min_id"], Id_info_dict["item_max_id"]],
        train_data_pair=data_pair_dict["train"],
    )
    pred_dataloader = DataLoader(all_pair_without_train, batch_size=100, shuffle=False)
    pred_list = []
    
    with torch.no_grad():
        for tensor in pred_dataloader:
            batch_model_input = utils.BatchGetModelInput(
                user_features_dict=features_dict_dict["user"],
                item_features_dict=features_dict_dict["item"],
                batch_user_id=tensor[:,0],
                batch_item_id=tensor[:,1],
                user_ftype=features_type_dict["user"],
                item_ftype=features_type_dict["item"],
            )
            
            preds = model.predict(batch_model_input, batch_size=256)
            pred_list.append(preds.reshape(-1))
            
    preds = np.concatenate(pred_list)
    
    recommend_item = all_pair_without_train[metrics.TopkRecommend(preds, k=10), 1]
    
    # Calculate metrics
    recall_ = metrics.Recall(metrics.GetUserRelevant(Id_info_dict["user_id"], data_pair_dict["test"]), recommend_item)
    ndcg_ = metrics.NDCG(metrics.GetUserRelevant(Id_info_dict["user_id"], data_pair_dict["test"]), recommend_item)
    return recall_, ndcg_