
import numpy as np

def IsRelevant(score, threshold=3.5):
    """
    according threshold to distinguish relevant/irrelevant item for specific user
    """
    one_hot_relevant = score >= threshold
    return one_hot_relevant

def GetUserRelevant(user_id, data, threshold=3.5):
    """
    get relevant item ID for specific user from dataset
    """
    filter_index = data[:,0] == user_id
    
    if (filter_index).sum() == 0:
        return np.array([]).astype(int)
    else:
        temp_data = data[filter_index,:]
        temp_data = temp_data[temp_data[:,1].argsort()] # use item id to sort
        is_relevent = IsRelevant(temp_data[:,2], threshold)
        return temp_data[is_relevent,1].astype(int)

def TopkRecommend(pred_score, k=10):
    """
    rank values of predicton
    pred_score: nd.array
    """
    indices = np.argsort(pred_score)[::-1][:k]
    return indices
#     values, indices = torch.topk(pred_score, k)
#     one_hot_indices = torch.zeros(pred_score.shape)
#     one_hot_indices[indices] = 1
#     return indices, one_hot_indices

def Recall(relevant_item_id, pred_item_id):
    """
    Calculate recall
    relevant_item_id: integer array
    pred_item_id: integer array
    """
    if len(relevant_item_id) > 0:
        return len(set(relevant_item_id) & set(pred_item_id)) / len(relevant_item_id)
    else:
        return 1

def DCG(relevant_item_id, pred_item_id):
    """
    Calculate DCG
    """
    indices = np.arange(0, len(pred_item_id)) + 1
    return np.sum(np.isin(pred_item_id, relevant_item_id).astype(int) / np.log2(indices + 1))

def IDCG(relevant_item_id, pred_item_id):
    """
    Calculate IDCG
    """
    indices = np.arange(0, len(pred_item_id)) + 1
    return np.sum(np.array([1] * len(pred_item_id)) / np.log2(indices + 1))

def NDCG(relevant_item_id, pred_item_id):
    """
    Calculate NDCG
    """
    return DCG(relevant_item_id, pred_item_id) / IDCG(relevant_item_id, pred_item_id)