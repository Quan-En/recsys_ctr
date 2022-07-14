import numpy as np
# import pandas as pd
import pickle

from itertools import product
# from functools import reduce
from torch.utils.data import Dataset

class RateDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]
    
    def __len__(self):
        return self.user_tensor.size(0)


def DelLowInteraction(iteraction_data, times=3):
    
    # Assume first column is User-id
    unique_user_id, id_counts = np.unique(iteraction_data[:,0], return_counts=True)

    # Target id
    need_del_id = unique_user_id[id_counts < times]
    if len(need_del_id) > 0: return np.delete(iteraction_data, np.isin(iteraction_data[:,0], need_del_id), axis=0)
    else: return iteraction_data

# def DelLowInteraction(iteraction_data, other_data_list, times=3):
    
#     # Assume first column is User-id
#     unique_user_id, id_counts = np.unique(iteraction_data[:,0], return_counts=True)

#     # Target id
#     need_del_id = unique_user_id[id_counts < 3]
#     if len(need_del_id) > 0:
#         return np.delete(iteraction_data, np.isin(iteraction_data[:,0], need_del_id), axis=0),\
#             [np.delete(data, np.isin(data[:,0], need_del_id), axis=0) for data in other_data_list],\
#             need_del_id
#     else:
#         return iteraction_data, other_data_list, need_del_id

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
    
def GetAllPair(user_id, item_id_range):
    """use numpy"""
    all_item_id = np.arange(item_id_range[0], item_id_range[1]+1)
    
    all_pair = np.array(list(product([user_id], all_item_id))).astype(int)
    
    return all_pair

def GetAllPairWithoutTrainSet(user_id, item_id_range, train_data_pair):
    """use numpy"""
    all_pair = GetAllPair(user_id, item_id_range)
    
    train_data_filter_index = train_data_pair[:,0] == user_id
    train_data_pair = train_data_pair[train_data_filter_index,:]
    train_item_id = np.sort(train_data_pair[:,1])
    
    return np.delete(arr=all_pair, obj=train_item_id-item_id_range[0], axis=0)

def AddNewFeatures2Dict(added_dict:dict, feature_name:str, keys, values):

    # Store empty list
    for key, value in added_dict.items(): added_dict[key][feature_name] = []

    # Append values to list
    for key, value in zip(keys, values): added_dict[key][feature_name] += [value]

    # Avoid empty list
    all_dict_keys = np.array(list(added_dict.keys())).astype(int)
    feature_empty_id = all_dict_keys[~np.isin(all_dict_keys, keys)]
    for key in feature_empty_id: added_dict[key][feature_name] += [0]

    return added_dict

def AppendFeaturesDict(appended_dict:dict, feature_name:str, keys, values, value_to_list=False):
    """
    Old version of AddNewFeatures2Dict
    """
    if value_to_list:
        for key, value in zip(keys, values):appended_dict[key][feature_name] = [value]
    else:
        for key, value in zip(keys, values):appended_dict[key][feature_name] = value
    return appended_dict

# def GetAllFeatures(features_dict, search_id):
#     return reduce(lambda x, y:x+y, features_dict[search_id].values())

def GetFeaturesList(features_dict, features_type, search_ids):
    feat_keys = features_type.keys()
    return [np.array([features_dict[int(id)][key] for id in search_ids]) for key in feat_keys]

# def BatchGetAllFeatures(user_features_dict, item_features_dict, batch_user_id, batch_item_id):
#     batch_features = [
#         GetAllFeatures(user_features_dict, int(uid)) + GetAllFeatures(item_features_dict, int(iid))
#         for uid, iid in zip(batch_user_id, batch_item_id)
#     ]
#     return batch_features

def BatchGetModelInput(user_features_dict, item_features_dict, batch_user_id, batch_item_id, user_ftype:dict, item_ftype:dict):

    feat_list = GetFeaturesList(user_features_dict, user_ftype, batch_user_id) + \
        GetFeaturesList(item_features_dict, item_ftype, batch_item_id)
    
    v_names_list = [k + "_" + str(i) for k, v in user_ftype.items() for i in range(v["f_len"])] + \
        [k + "_" + str(i) for k, v in item_ftype.items() for i in range(v["f_len"])]
    
    return {v_name:data_arr for v_name, data_arr in zip(v_names_list, feat_list)}

# def BatchGetModelInput(user_features_dict, item_features_dict, batch_user_id, batch_item_id, user_ftype:dict, item_ftype:dict):
#     data_arr = np.array(BatchGetAllFeatures(user_features_dict, item_features_dict, batch_user_id, batch_item_id))
    
#     v_names_list = [k + "_" + str(i) for k, v in user_ftype.items() for i in range(v["f_len"])] + \
#         [k + "_" + str(i) for k, v in item_ftype.items() for i in range(v["f_len"])]
    
#     return {v_name:data_arr[:,idx] for idx, v_name in enumerate(v_names_list)}

# def AllFeatures2DataFrame(data_list, user_ftype:dict, item_ftype:dict):
#     df = pd.DataFrame(data_list)
    
#     v_names_list = [k + "_" + str(i) for k, v in user_ftype.items() for i in range(v["f_len"])] + \
#         [k + "_" + str(i) for k, v in item_ftype.items() for i in range(v["f_len"])]
#     df.columns = v_names_list
#     return df

# def BatchFeatures2DataFrame(u, i, user_features_dict, item_features_dict, user_ftype, item_ftype):
#     data_list = BatchGetAllFeatures(user_features_dict, item_features_dict, u, i)
#     data_df = AllFeatures2DataFrame(data_list, user_ftype, item_ftype)
#     return data_df

def features_type(f_type:str="Sparse", f_len:int=1, vocabulary_size=None, max_len=None):
    """
    f_type: Sparse / VarLenSparse / Dense
    f_len: how many columns
    max_len: For VarLenSparse, maximum of label co-occur
    """
    return dict(f_type=f_type, f_len=f_len, vocabulary_size=vocabulary_size, max_len=max_len)

def SaveObject(name, obj):
    with open(name, "wb+") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def LoadObject(name):
    """Load object from specific path (.pkl)"""
    with open(name, "rb") as f:
        return pickle.load(f)