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
train_batch_size = 64
val_batch_size = 256
# Init dataset
training_dir = "../processed_dataset/training/"
training_set = AuxiliaryDataset(data_dir=training_dir, dataset_name="training", forecast_horizon=forecast_horizon)
training_set_dataloader = DataLoader(dataset=training_set, batch_size=train_batch_size, shuffle=True)
# Validation 2011
validation_2011_dir = "../processed_dataset/validation_2011/"
validation_2011 = AuxiliaryDataset(data_dir=validation_2011_dir, dataset_name="validation_2011", forecast_horizon=forecast_horizon)
validation_2011_dataloader = DataLoader(dataset=validation_2011, batch_size=val_batch_size, shuffle=False)
# Validation 2012
validation_2012_dir = "../processed_dataset/validation_2012/"
validation_2012 = AuxiliaryDataset(data_dir=validation_2012_dir, dataset_name="validation_2012", forecast_horizon=forecast_horizon)
validation_2012_dataloader = DataLoader(dataset=validation_2012, batch_size=val_batch_size, shuffle=False)
# Test 2015
# test_2015_dir = "../processed_dataset/test_2015/"
# test_2015 = SkyImagesAuxiliaryDataset(data_dir=test_2015_dir, dataset_name="test_2015", forecast_horizon=forecast_horizon)
# test_2015_dataloader = DataLoader(dataset=test_2015, batch_size=batch_size, shuffle=False)
# # # Test 2016
# test_2016_dir = "../processed_dataset/test_2016/"
# test_2016 = SkyImagesAuxiliaryDataset(data_dir=test_2016_dir, dataset_name="test_2016", forecast_horizon=forecast_horizon)
# test_2016_dataloader = DataLoader(dataset=test_2016, batch_size=batch_size, shuffle=False)


# ***************************************************************************************************************
# ***************************************************************************************************************
# ******************************* Load model from checkpoint and init parameters ********************************
# ***************************************************************************************************************
# ***************************************************************************************************************
# Init auxiliary_lstm
auxiliary_indim = 7
auxiliary_nb_layer = 5
auxiliary_hid_dim = 128
auxiliary_outdim = 3
auxiliary_lstm = Auxiliary_LSTM(auxiliary_nb_layer, auxiliary_indim, auxiliary_hid_dim, auxiliary_outdim)
# auxiliary_lstm = torch.nn.DataParallel(auxiliary_lstm)
auxiliary_lstm.to(device=device)

# Init optim
optim = torch.optim.Adam(params=auxiliary_lstm.parameters(), lr=0.0005)

# Init loss
l2_loss = torch.nn.MSELoss()
# Init some variables for training
train_loss = []
val_loss = []
nb_epochs = 100

for epoch in range(nb_epochs):
    print(f"Epoch {epoch}")
    if epoch == 15:
        for g in optim.param_groups:
            g['lr'] = 0.00025
    if epoch == 20:
        for g in optim.param_groups:
            g['lr'] = 0.0001

    # Training
    auxiliary_lstm.train()
    epoch_train_loss = 0
    for i, sample in enumerate(tqdm(training_set_dataloader)):
        id_list, auxiliary, target_auxiliary = sample
        auxiliary = auxiliary.to(device=device).float()
        target_auxiliary = target_auxiliary.to(device=device).float()
        target_auxiliary = torch.cat([auxiliary[:,1:], target_auxiliary], dim=1)
        auxiliary_pred = auxiliary_lstm(auxiliary, total_length=target_auxiliary.shape[1],  device=device)
        loss = l2_loss(auxiliary_pred[:,:,-3:], target_auxiliary[:,:,-3:])
        optim.zero_grad()
        loss.backward()
        optim.step()
        epoch_train_loss += loss.item()*len(target_auxiliary)
    train_loss.append(epoch_train_loss/len(training_set))

    # Validation
    auxiliary_lstm.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for validation_set in [validation_2011_dataloader, validation_2012_dataloader]:
            for i, sample in enumerate(tqdm(validation_set)):
                id_list, auxiliary, target_auxiliary = sample
                auxiliary = auxiliary.to(device=device).float()
                target_auxiliary = target_auxiliary.to(device=device).float()
                target_auxiliary = torch.cat([auxiliary[:,1:], target_auxiliary], dim=1)
                auxiliary_pred = auxiliary_lstm(auxiliary, total_length=target_auxiliary.shape[1], device=device)
                loss = l2_loss(auxiliary_pred[:,:,-3:], target_auxiliary[:,:,-3:])

                epoch_val_loss += loss.item()*len(target_auxiliary)
        val_loss.append(epoch_val_loss/ (len(validation_2011) + len(validation_2012)))

    print(f"Finish epoch {epoch}. Train_loss = {train_loss[-1]}. Val_loss = {val_loss[-1]}")

    # Save model
    save_path = f"../checkpoint/auxiliary_lstm_{forecast_horizon}/auxiliary_lstm_{epoch}epoch.pt"
    save_model(epoch=epoch, model=auxiliary_lstm, optim=optim, train_loss=train_loss, val_loss=val_loss, save_path=save_path)
    # Save learning curve
    learning_curve_save_path = f"../checkpoint/auxiliary_lstm_{forecast_horizon}/learning_curve_{epoch}epoch"
    save_learning_curve(train_loss, val_loss, learning_curve_save_path)

