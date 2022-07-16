import numpy as np

import torch
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from utils import utils, load, deepctr_utils

from model import DeepCTR_Model

import argparse

def main():
    data_name_list = ["movielens", "yelp", "douban_book"]
    model_name_list = ["deepfm", "ipnn", "opnn", "pin", "ccpm", "afm", "nfm", "xdeepfm", "fnn", "wd", "dcn"]
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", "-d", type=str, nargs='?', choices=data_name_list, help="DataSet name")
    parser.add_argument("--modelname", "-m", type=str, nargs='?', choices=model_name_list, help="Model name")
    args = parser.parse_args()
    
    # Load dataset and load feature's type dict
    dataset = load.LoadDataSet(args.dataname)
    
    user_item = dataset["user_item"]
    all_uids = dataset["uids"]
    all_iids = dataset["iids"]
    user_features_dict, item_features_dict = dataset["user_features_dict"], dataset["item_features_dict"]
    user_features_type, item_features_type = dataset["user_features_type"], dataset["item_features_type"]
    
    # Get user's feature columns and item's feature columns
    features_columns = deepctr_utils.GetFeatColumns(user_features_type) + deepctr_utils.GetFeatColumns(item_features_type)
    
    train_data, test_data = train_test_split(user_item, test_size=0.2, random_state=1024)
    train_score = train_data[:,2]
    
    train_model_input = utils.BatchGetModelInput(
        user_features_dict=user_features_dict,
        item_features_dict=item_features_dict,
        batch_user_id=train_data[:,0].astype(int),
        batch_item_id=train_data[:,1].astype(int),
        user_ftype=user_features_type,
        item_ftype=item_features_type,
    )
    
    train_dataset = utils.RateDataset(
        user_tensor=torch.LongTensor(train_data[:,0].astype(int)),
        item_tensor=torch.LongTensor(train_data[:,1].astype(int)),
        target_tensor=torch.Tensor(train_data[:,2]),
    )

    test_dataset = utils.RateDataset(
        user_tensor=torch.LongTensor(test_data[:,0].astype(int)),
        item_tensor=torch.LongTensor(test_data[:,1].astype(int)),
        target_tensor=torch.Tensor(test_data[:,2]),
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    
    model = DeepCTR_Model.BuildModel(args.modelname, features_columns, device)
    model.compile("adam", "mse", metrics=['mse'], )
    
    history = model.fit(train_model_input, train_score, batch_size=256, epochs=20, verbose=2)
    
    total_loss = 0
    print("Calculating TrainSet RMSE...")
    with torch.no_grad():
        for batch in tqdm(train_loader):
            u, i, r = batch[0], batch[1], batch[2]
            r = r.float()

            # forward pass
            batch_model_input = utils.BatchGetModelInput(
                user_features_dict=user_features_dict,
                item_features_dict=item_features_dict,
                batch_user_id=u,
                batch_item_id=i,
                user_ftype=user_features_type,
                item_ftype=item_features_type,
            )

            preds = model.predict(batch_model_input, batch_size=256)
            loss = np.sum((preds.reshape(-1) - r.numpy())**2).item()
            total_loss += loss
        total_loss /= train_dataset.__len__()
    print("TrainSet RMSE:", np.sqrt(total_loss))
    
    total_loss = 0
    print("Calculating TestSet RMSE...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            u, i, r = batch[0], batch[1], batch[2]
            r = r.float()

            # forward pass
            batch_model_input = utils.BatchGetModelInput(
                user_features_dict=user_features_dict,
                item_features_dict=item_features_dict,
                batch_user_id=u,
                batch_item_id=i,
                user_ftype=user_features_type,
                item_ftype=item_features_type,
            )

            preds = model.predict(batch_model_input, batch_size=256)
            loss = np.sum((preds.reshape(-1) - r.numpy())**2).item()

            total_loss += loss
        total_loss /= train_dataset.__len__()
    print("TestSet RMSE:", np.sqrt(total_loss))
    
    all_test_data_id = np.unique(test_data[:,0]).astype(int)
    
    metrics_list = []
    for uid in tqdm(all_test_data_id):
        metrics_list.append(deepctr_utils.Predict(
            Id_info_dict=dict(user_id=uid, item_min_id=all_iids[0], item_max_id=all_iids[-1]),
            data_pair_dict=dict(train=train_data[:,:2].astype(int), test=test_data.astype(int)),
            features_dict_dict=dict(user=user_features_dict, item=item_features_dict),
            features_type_dict=dict(user=user_features_type, item=item_features_type),
            model=model,
        ))

    metrics_arr = np.mean(metrics_list, axis=0)
    print("Mean of Recall@10:", metrics_arr[0])
    print("Mean of NDCG@10:", metrics_arr[1])
        
    utils.SaveObject("metric_result/" + args.modelname + "_" + args.dataname + "_metrics.pkl", metrics_list)
    
if __name__ == "__main__":
    main()