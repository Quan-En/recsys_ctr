import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import utils, metrics

def Predict(Id_info_dict, data_pair_dict, model, device):
    
    all_pair_without_train = utils.GetAllPairWithoutTrainSet(
        user_id=Id_info_dict["user_id"],
        item_id_range=[Id_info_dict["item_min_id"], Id_info_dict["item_max_id"]],
        train_data_pair=data_pair_dict["train"],
    )
    pred_dataloader = DataLoader(all_pair_without_train, batch_size=100, shuffle=False)
    pred_list = []
    
    with torch.no_grad():
        for tensor in pred_dataloader:           
            preds = model(tensor[:,0].to(device), tensor[:,1].to(device))
            pred_list.append(preds.reshape(-1).cpu().numpy())
            
    preds = np.concatenate(pred_list)
    
    recommend_item = all_pair_without_train[metrics.TopkRecommend(preds, k=10), 1]
    
    # Calculate metrics
    recall_ = metrics.Recall(metrics.GetUserRelevant(Id_info_dict["user_id"], data_pair_dict["test"]), recommend_item)
    ndcg_ = metrics.NDCG(metrics.GetUserRelevant(Id_info_dict["user_id"], data_pair_dict["test"]), recommend_item)
    return recall_, ndcg_