

import numpy as np
from utils import utils

data_name_dict = dict(
    movielens=dict(
        user=["user_age", "user_occupation"],
        item=["movie_genre"],
        user_item="user_movie",
    ),
    yelp=dict(
        user=["user_compliment"],
        item=["business_city", "business_category"],
        user_item="user_business",
    ),
    douban_book=dict(
        user=["user_location", "user_group"],
        item=["book_year", "book_author", "book_publisher"],
        user_item="user_book",
    ),
)

def GetIDsRange(user_data_list, item_data_list, user_item):
    user_min_id = int(min([user_item[:,0].min()] + [u_data[:,0].min() for u_data in user_data_list]))
    user_max_id = int(max([user_item[:,0].max()] + [u_data[:,0].max() for u_data in user_data_list]))

    item_min_id = int(min([user_item[:,1].min()] + [i_data[:,0].min() for i_data in item_data_list]))
    item_max_id = int(max([user_item[:,1].max()] + [i_data[:,0].max() for i_data in item_data_list]))
    
    print("user'ID: (Min, Max)=({}, {})".format(user_min_id, user_max_id))
    print("item'ID: (Min, Max)=({}, {})".format(item_min_id, item_max_id))
    
    all_uids = np.arange(user_min_id, user_max_id+1)
    all_iids = np.arange(item_min_id, item_max_id+1)
    
    return all_uids, all_iids

def LoadDataSet(dataname:str):
    
    print("Load dataset: " + dataname, "\n")
    
    print("Loading...", end = "")
    user_data_name_list = data_name_dict[dataname]["user"]
    item_data_name_list = data_name_dict[dataname]["item"]
    user_item_name = data_name_dict[dataname]["user_item"]
    
    user_data_list = [
        np.genfromtxt("data/"+ dataname +"/" + fname + ".dat")
        for fname in user_data_name_list
    ]
    
    item_data_list = [
        np.genfromtxt("data/"+ dataname +"/" + fname + ".dat")
        for fname in item_data_name_list
    ]
    
    user_item = np.genfromtxt("data/" + dataname + "/" + user_item_name + ".dat")
    print("[complete]")
    
    print("Deleting low interaction...", end = "")
    user_item = utils.DelLowInteraction(user_item, times=3)
    print("[complete]")
    all_uids, all_iids = GetIDsRange(user_data_list, item_data_list, user_item)
    
    print("Collecting (user/item) features dictionary...", end = "")
    user_features_dict, item_features_dict = CollectDataFeat(
        dataname=dataname,
        ids_list=[all_uids, all_iids],
        user_data_list=user_data_list,
        item_data_list=item_data_list,
    )
    print("[complete]")
    
    print("Collecting (user/item) features type dictionary...", end = "")
    user_features_type, item_features_type = CollectFeatType(dataname)
    print("[complete]")

    print("Padding VarLenSparse features...", end = "")
    for feat_key, feat_value in user_features_type.items():
        if feat_value["f_type"] == "VarLenSparse":
            for user_key, user_value in user_features_dict.items():
                paded_len = feat_value["max_len"] - len(user_value[feat_key])
                user_features_dict[user_key][feat_key] += [0] * paded_len

    for feat_key, feat_value in item_features_type.items():
        if feat_value["f_type"] == "VarLenSparse":
            for item_key, item_value in item_features_dict.items():
                paded_len = feat_value["max_len"] - len(item_value[feat_key])
                item_features_dict[item_key][feat_key] += [0] * paded_len
    print("[complete]")

    dataset_dict = dict(
        user_data_list=user_data_list,
        item_data_list=item_data_list,
        user_item=user_item,
        uids=all_uids,
        iids=all_iids,
        user_features_dict=user_features_dict,
        item_features_dict=item_features_dict,
        user_features_type=user_features_type,
        item_features_type=item_features_type,
    )
    
    return dataset_dict

def CollectDataFeat(dataname:str, ids_list:list, user_data_list:list, item_data_list:list):
    """
    Collect user's features dict and utem's features dict
    ids_list: user ids, item ids
    """
    user_features_dict = {i:dict() for i in ids_list[0]}
    item_features_dict = {i:dict() for i in ids_list[1]}
    
    if dataname == "movielens":
        user_short_feat_name_list = ["age", "occ"]
        item_short_feat_name_list = ["genre"] # genre: Multi-values
    elif dataname == "yelp":
        user_short_feat_name_list = ["compliment"] # compliment: Multi-values
        item_short_feat_name_list = ["city", "category"] # category: Multi-values
    elif dataname == "douban_book":
        user_short_feat_name_list = ["location", "group"] # group: Multi-values
        item_short_feat_name_list = ["year", "author", "publisher"]

    for idx, data_arr in enumerate(user_data_list):
        user_features_dict = utils.AddNewFeatures2Dict(
            added_dict=user_features_dict,
            feature_name=user_short_feat_name_list[idx],
            keys=data_arr[:,0].astype(int),
            values=data_arr[:,1].astype(int),
        )
    for idx, data_arr in enumerate(item_data_list):
        item_features_dict = utils.AddNewFeatures2Dict(
            added_dict=item_features_dict,
            feature_name=item_short_feat_name_list[idx],
            keys=data_arr[:,0].astype(int),
            values=data_arr[:,1].astype(int),
        )
    return user_features_dict, item_features_dict

def CollectFeatType(dataname:str):
    if dataname == "movielens":
        user_features_type = dict(
            age=utils.features_type("Sparse", 1, 9),
            occ=utils.features_type("Sparse", 1, 22),
        )
        item_features_type = dict(
            genre=utils.features_type("VarLenSparse", 1, 19, 6),
        )
    elif dataname == "yelp":
        user_features_type = dict(
            compliment=utils.features_type("VarLenSparse", 1, 12, 11),
        )
        item_features_type = dict(
            city=utils.features_type("Sparse", 1, 48),
            category=utils.features_type("VarLenSparse", 1, 512, 10),
        )
    elif dataname == "douban_book":
        user_features_type = dict(
            location=utils.features_type("Sparse", 1, 454),
            group=utils.features_type("VarLenSparse", 1, 2937, 629),
        )
        item_features_type = dict(
            year=utils.features_type("Sparse", 1, 65),
            author=utils.features_type("Sparse", 1, 10806),
            publisher=utils.features_type("Sparse", 1, 1816),
        )
    return user_features_type, item_features_type