import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from utils import utils, load, fm_utils

from model import FactorizationMachine

import argparse

def main():
    data_name_list = ["movielens", "yelp", "douban_book"]
    
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", "-d", type=str, nargs='?', choices=data_name_list, help="DataSet name")
    args = parser.parse_args()

    dataset = load.LoadDataSet(args.dataname)

    user_item = dataset["user_item"]
    all_uids = dataset["uids"]
    all_iids = dataset["iids"]
    user_features_dict, item_features_dict = dataset["user_features_dict"], dataset["item_features_dict"]
    user_features_type, item_features_type = dataset["user_features_type"], dataset["item_features_type"]

    train_data, test_data = train_test_split(user_item, test_size=0.2, random_state=1024)

    train_dataset = utils.RateDataset(
        user_tensor=torch.LongTensor(train_data[:,0].astype(int)),
        item_tensor=torch.LongTensor(train_data[:,1].astype(int)),
        target_tensor=torch.LongTensor(train_data[:,2]),
    )

    test_dataset = utils.RateDataset(
        user_tensor=torch.LongTensor(test_data[:,0].astype(int)),
        item_tensor=torch.LongTensor(test_data[:,1].astype(int)),
        target_tensor=torch.LongTensor(test_data[:,2]),
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


    total_dim = 0
    for k, v in user_features_type.items():
        if "Sparse" in v["f_type"]:
            total_dim += (v["vocabulary_size"] - 1)
        else:
            total_dim += 1
    for k, v in item_features_type.items():
        if "Sparse" in v["f_type"]:
            total_dim += (v["vocabulary_size"] - 1)
        else:
            total_dim += 1
    
    my_model = FactorizationMachine.Model(field_dims=total_dim, embed_dim=10).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(my_model.parameters(), lr=0.01)

    for epoch in range(20):        
        total_train_loss = 0
        total_test_loss = 0

        # Training stage
        my_model.train()
        for batch in tqdm(train_loader):
            u, i, r = batch[0], batch[1], batch[2]
            r = r.float()
            
            # forward pass
            features = utils.GetFeatArray(user_features_dict, item_features_dict, u, i, user_features_type, item_features_type)
            features = torch.Tensor(features).to(device)
            preds = my_model(features)
            loss = criterion(preds, r.to(device))
            total_train_loss += loss.item()

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluating stage
        my_model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                u, i, r = batch[0], batch[1], batch[2]
                r = r.float()

                # forward pass
                features = utils.GetFeatArray(user_features_dict, item_features_dict, u, i, user_features_type, item_features_type)
                features = torch.Tensor(features).to(device)
                preds = my_model(features)
                loss = criterion(preds, r.to(device))
                total_test_loss += loss.item()

        print("Epoch [{}/20], TrainLoss: {:.4f}, TestLoss: {:.4f}".format(epoch + 1, total_train_loss, total_test_loss))
    
    my_model.eval()

    total_loss = 0
    print("Calculating TrainSet RMSE...")
    with torch.no_grad():
        for batch in tqdm(train_loader):
            u, i, r = batch[0], batch[1], batch[2]
            r = r.float()

            # forward pass
            features = utils.GetFeatArray(user_features_dict, item_features_dict, u, i, user_features_type, item_features_type)
            features = torch.Tensor(features).to(device)

            preds = my_model(features)
            loss = np.sum((preds.reshape(-1).cpu().numpy() - r.numpy())**2).item()
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
            features = utils.GetFeatArray(user_features_dict, item_features_dict, u, i, user_features_type, item_features_type)
            features = torch.Tensor(features).to(device)

            preds = my_model(features)
            loss = np.sum((preds.reshape(-1).cpu().numpy() - r.numpy())**2).item()

            total_loss += loss
        total_loss /= train_dataset.__len__()
    print("TestSet RMSE:", np.sqrt(total_loss))


    all_test_data_id = np.unique(test_data[:,0]).astype(int)

    metrics_list = []
    for uid in tqdm(all_test_data_id):
        metrics_list.append(fm_utils.Predict(
            Id_info_dict=dict(user_id=uid, item_min_id=all_iids[0], item_max_id=all_iids[-1]),
            data_pair_dict=dict(train=train_data[:,:2].astype(int), test=test_data.astype(int)),
            features_dict_dict=dict(user=user_features_dict, item=item_features_dict),
            features_type_dict=dict(user=user_features_type, item=item_features_type),
            model=my_model,
            device=device,
        ))
    
    metrics_arr = np.mean(metrics_list, axis=0)
    print("Mean of Recall@10:", metrics_arr[0])
    print("Mean of NDCG@10:", metrics_arr[1])

    utils.SaveObject("metric_result/fm_" + args.dataname + "_metrics.pkl", metrics_list)

if __name__ == "__main__":
    main()