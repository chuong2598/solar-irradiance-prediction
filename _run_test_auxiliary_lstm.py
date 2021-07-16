import os
import torch
import torchvision
import numpy as np
# from piq import ssim, psnr
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from utils import *
from dataset import AuxiliaryDataset
from models import Auxiliary_LSTM

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(suppress=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
# print("Device: {} GPUs - {}".format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
print()

# ***************************************************************************************************************
# ************************************************* Init dataset ************************************************
# ***************************************************************************************************************
# ***************************************************************************************************************
# Define forecast horizon
forecast_horizon = "1h"
val_batch_size = 4096
# Test 2015
test_2015_dir = "../processed_dataset/test_2015/"
test_2015 = AuxiliaryDataset(data_dir=test_2015_dir, dataset_name="test_2015", forecast_horizon=forecast_horizon)
test_2015_dataloader = DataLoader(dataset=test_2015, batch_size=val_batch_size, shuffle=False)
# Test 2016
test_2016_dir = "../processed_dataset/test_2016/"
test_2016 = AuxiliaryDataset(data_dir=test_2016_dir, dataset_name="test_2016", forecast_horizon=forecast_horizon)
test_2016_dataloader = DataLoader(dataset=test_2016, batch_size=val_batch_size, shuffle=False)


# ***************************************************************************************************************
# ***************************************************************************************************************
# ******************************* Load model from checkpoint and init parameters ********************************
# ***************************************************************************************************************
# ***************************************************************************************************************
forecast_horizon = "1h"
auxiliary_lstm_load_epoch = 49
# Init auxiliary_lstm
auxiliary_nb_layer = 5
auxiliary_indim = 7
auxiliary_hid_dim = 128
auxiliary_outdim = 3
auxiliary_lstm = Auxiliary_LSTM(auxiliary_nb_layer, auxiliary_indim, auxiliary_hid_dim, auxiliary_outdim)
auxiliary_lstm.to(device=device)
# Load auxiliary_lstm
auxiliary_lstm_checkpoint_path = f"../checkpoint/auxiliary_lstm_{forecast_horizon}/auxiliary_lstm_{auxiliary_lstm_load_epoch}epoch.pt"
auxiliary_lstm_checkpoint = torch.load(auxiliary_lstm_checkpoint_path)
auxiliary_lstm.load_state_dict(auxiliary_lstm_checkpoint["model_state_dict"])
# auxiliary_lstm = torch.nn.DataParallel(auxiliary_lstm)
del auxiliary_lstm_checkpoint



# Init loss
l2_loss = torch.nn.MSELoss()
# Init some variables for training


# Validation
pred = []
truth = []
auxiliary_lstm.eval()
epoch_val_loss = 0
with torch.no_grad():
    for validation_set in [test_2015_dataloader, test_2016_dataloader]:
        for i, sample in enumerate(tqdm(validation_set)):
            # PredRNN Prediction
            id_list, auxiliary, target_auxiliary = sample
            auxiliary = auxiliary.to(device=device).float()
            target_auxiliary = target_auxiliary.to(device=device).float()
            target_auxiliary = torch.cat([auxiliary[:,1:], target_auxiliary], dim=1)
            auxiliary_pred = auxiliary_lstm(auxiliary, total_length=target_auxiliary.shape[1], device=device)
            loss = l2_loss(auxiliary_pred[:,:,-3:], target_auxiliary[:,:,-3:])
            # if i != len(validation_set) - 1:
            pred.append(auxiliary_pred)
            truth.append(target_auxiliary)

            epoch_val_loss += loss.item()*len(target_auxiliary)
    print("Average l1 error: ", epoch_val_loss/ (len(test_2015) + len(test_2016)))

pred = torch.cat(pred, dim=0)
truth = torch.cat(truth, dim=0)
# Denomalize
mean, std = get_auxiliary_stats()
mean = torch.tensor(mean[[0,6,7]]).to(device=device)
std = torch.tensor(std[[0,6,7]]).to(device=device)
pred[:,:,-3:] = pred[:,:,-3:]*std + mean
truth[:,:,-3:] = truth[:,:,-3:]*std + mean
# Print l1 error
l1_error = torch.mean(torch.abs(pred-truth), axis=0).detach().cpu().numpy()
print("l1 error in each timestep:")
print(l1_error)


