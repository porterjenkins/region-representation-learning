import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
from sklearn.model_selection import KFold
from experiments.prediction import HousePriceModel, TrafficVolumeModel


if len(sys.argv) == 1:
    raise Exception("User must input task: {'house_price' or 'traffic'")
else:
   task = sys.argv[1]
   n_epochs = int(sys.argv[2])

n_folds = 5

print("K-Fold Prediction - training epochs: {}".format(n_epochs))



c = get_config()
region_grid = RegionGrid(config=c)
region_grid.load_weighted_mtx()

if task == 'house_price':
    input_data = region_grid.load_housing_data(c['housing_data_file'])
    # Initialize Models
    naive_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs)
    deepwalk_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['deepwalk_file'])
    re_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'])
    joint_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'], c['deepwalk_file'])
    nmf_mod = HousePriceModel(region_grid.idx_coor_map, c, n_epochs, c['nmf_file'])

elif task == 'traffic':
    input_data = region_grid.load_traffic_data(c['traffic_data_file'])
    # Initialize Models
    naive_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs)
    deepwalk_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['deepwalk_file'])
    re_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'])
    joint_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['embedding_file'], c['deepwalk_file'])
    nmf_mod = TrafficVolumeModel(region_grid.idx_coor_map, c, n_epochs, c['nmf_file'])


else:
    raise NotImplementedError("User must input task: {'house_price' or 'traffic'")



# Get Features
naive_mod.get_features(input_data)
deepwalk_mod.get_features(input_data)
nmf_mod.get_features(input_data)
re_mod.get_features(input_data)
joint_mod.get_features(input_data)


k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=1990)

naive_err = np.zeros((n_folds, 2))
deepwalk_err = np.zeros((n_folds, 2))
embed_err = np.zeros((n_folds, 2))
nmf_err = np.zeros((n_folds, 2))
deepwalk_and_proposed_err = np.zeros((n_folds, 2))


train_ind_arr = np.arange(deepwalk_mod.X.shape[0])

fold_cntr = 0
for train_idx, test_idx in k_fold.split(train_ind_arr):
    print("Beginning Fold: {}".format(fold_cntr+1))
    # Naive Model
    rmse, mae = naive_mod.train_eval(train_idx, test_idx)
    naive_err[fold_cntr, 0] = rmse
    naive_err[fold_cntr, 1] = mae


    # DeepWalk Model
    rmse, mae = deepwalk_mod.train_eval(train_idx, test_idx)
    deepwalk_err[fold_cntr, 0] = rmse
    deepwalk_err[fold_cntr, 1] = mae

    # Matrix Factorization Model
    rmse, mae = nmf_mod.train_eval(train_idx, test_idx)
    nmf_err[fold_cntr, 0] = rmse
    nmf_err[fold_cntr, 1] = mae


    # Embedding model
    rmse, mae = re_mod.train_eval(train_idx, test_idx)
    embed_err[fold_cntr, 0] = rmse
    embed_err[fold_cntr, 1] = mae

    # Joint Model
    rmse, mae = joint_mod.train_eval(train_idx, test_idx)
    deepwalk_and_proposed_err[fold_cntr, 0] = rmse
    deepwalk_and_proposed_err[fold_cntr, 1] = mae



    fold_cntr += 1

naive_err_mean = np.mean(naive_err, axis=0)
naive_err_std = np.std(naive_err, axis=0)

embed_err_mean = np.mean(embed_err, axis=0)
embed_err_std = np.std(embed_err, axis=0)

deepwalk_err_mean = np.mean(deepwalk_err, axis=0)
deepwalk_err_std = np.std(deepwalk_err, axis=0)

nmf_err_mean = np.mean(nmf_err, axis=0)
nmf_err_std = np.std(nmf_err, axis=0)

deepwalk_and_proposed_mean = np.mean(deepwalk_and_proposed_err, axis=0)
deepwalk_and_proposed_std = np.std(deepwalk_and_proposed_err, axis=0)


print("Naive Model:")
print('RMSE: {:.4f} ({:.4f})'.format(naive_err_mean[0], naive_err_std[0]))
print('MAE: {:.4f} ({:.4f})'.format(naive_err_mean[1], naive_err_std[1]))

print("Deepwalk Model:")
print('RMSE: {:.4f} ({:.4f})'.format(deepwalk_err_mean[0], deepwalk_err_std[0]))
print('MAE: {:.4f} ({:.4f})'.format(deepwalk_err_mean[1], deepwalk_err_std[1]))

print("Matrix Factorization Model:")
print('RMSE: {:.4f} ({:.4f})'.format(nmf_err_mean[0], nmf_err_std[0]))
print('MAE: {:.4f} ({:.4f})'.format(nmf_err_mean[1], nmf_err_std[1]))

print("Embedding Model:")
print('RMSE: {:.4f} ({:.4f})'.format(embed_err_mean[0], embed_err_std[0]))
print('MAE: {:.4f} ({:.4f})'.format(embed_err_mean[1], embed_err_std[1]))

print("Deepwalk + Proposed Model:")
print('RMSE: {:.4f} ({:.4f})'.format(deepwalk_and_proposed_mean[0], deepwalk_and_proposed_std[0]))
print('MAE: {:.4f} ({:.4f})'.format(deepwalk_and_proposed_mean[1], deepwalk_and_proposed_std[1]))
