import os
import torch
import torchvision
import numpy as np
# from piq import ssim, psnr
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from utils import *
from dataset import SolarPowerDataset
from models import Alex_Net_Auxiliary

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
print("Device: {} GPUs - {}".format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
print()


# ***************************************************************************************************************
# ***************************************************************************************************************
# ************************************************* Init dataset ************************************************
# ***************************************************************************************************************
# ***************************************************************************************************************
train_batch_size = 64
val_batch_size = 1024
# Init dataset
training_dir = "../processed_dataset/training/"
training_set = SolarPowerDataset(samples_dir=training_dir, dataset_name="training")
training_set_dataloader = DataLoader(dataset=training_set, batch_size=train_batch_size, shuffle=True)
# Validation 2011
validation_2011_dir = "../processed_dataset/validation_2011/"
validation_2011 = SolarPowerDataset(samples_dir=validation_2011_dir, dataset_name="validation_2011")
validation_2011_dataloader = DataLoader(dataset=validation_2011, batch_size=val_batch_size, shuffle=False)
# Validation 2012
validation_2012_dir = "../processed_dataset/validation_2012/"
validation_2012 = SolarPowerDataset(samples_dir=validation_2012_dir, dataset_name="validation_2012")
validation_2012_dataloader = DataLoader(dataset=validation_2012, batch_size=val_batch_size, shuffle=False)
# Test 2015
# test_2015_dir = "../processed_dataset/test_2015/"
# test_2015 = SolarPowerDataset(samples_dir=test_2015_dir, dataset_name="test_2015")
# test_2015_dataloader = DataLoader(dataset=test_2015, batch_size=batch_size, shuffle=False)
# # Test 2016
# test_2016_dir = "../processed_dataset/test_2016/"
# test_2016 = SolarPowerDataset(samples_dir=test_2016_dir, dataset_name="test_2016")
# test_2016_dataloader = DataLoader(dataset=test_2016, batch_size=batch_size, shuffle=False)



# ***************************************************************************************************************
# ***************************************************************************************************************
# ****************************************** Init model and parameters ******************************************
# ***************************************************************************************************************
# ***************************************************************************************************************
# Init model 
auxiliary_indim = 7
alex_net = Alex_Net_Auxiliary(auxiliary_indim)
# alex_net = torch.nn.DataParallel(alex_net)
alex_net.to(device=device)
optim = torch.optim.Adam(params=alex_net.parameters(), lr=0.002)


# Init loss
l1_loss = torch.nn.L1Loss()
# Init some variables for training
train_loss = []
train_nMAP = []
val_loss = []
val_nMAP = []
current_epoch = 0
nb_epochs = 100

for epoch in range(nb_epochs):
    if (epoch in [3, 10, 20]):
        for g in optim.param_groups:
            g['lr'] = g['lr'] / 2

    print(f"Start epoch {epoch}")
    epoch_train_loss = 0
    epoch_train_pred = []
    epoch_train_truth = []
    epoch_train_nMAP = 0
    alex_net.train()
    for i, (image_id, image, auxiliary, solar_power) in enumerate(tqdm(training_set_dataloader)):
        image = (image/255.0).to(device=device).permute(0,3,1,2)
        # print(image.shape)
        auxiliary = auxiliary.float().to(device=device)
        solar_power = solar_power.float().to(device=device)
        pred = alex_net(image, auxiliary).flatten()
        loss = l1_loss(pred, solar_power)
        optim.zero_grad()
        loss.backward()
        optim.step()
        epoch_train_loss += loss.item()*len(solar_power)
        epoch_train_pred += list(pred.cpu().detach().numpy())
        epoch_train_truth += list(solar_power.cpu().detach().numpy())

    train_loss.append(epoch_train_loss/len(training_set))
    epoch_train_pred = torch.tensor(epoch_train_pred)
    epoch_train_truth = torch.tensor(epoch_train_truth)
    epoch_train_nMAP = torch.mean(torch.abs(epoch_train_pred-epoch_train_truth)/torch.mean(epoch_train_truth)*100)
    train_nMAP.append(epoch_train_nMAP.item())

    alex_net.eval()
    epoch_val_loss = 0
    epoch_val_nMAP = 0
    epoch_val_pred = []
    epoch_val_truth = []
    with torch.no_grad():
        for val_i, validation_loader in enumerate([validation_2011_dataloader, validation_2012_dataloader]):
            for i, (image_id, image, auxiliary, solar_power) in enumerate(tqdm(validation_loader)):
                image = (image/255.0).to(device=device).permute(0,3,1,2)
                auxiliary = auxiliary.float().to(device=device)
                solar_power = solar_power.float().to(device=device)
                pred = alex_net(image, auxiliary).flatten()
                loss = l1_loss(pred, solar_power)
                epoch_val_loss += loss.item()*len(solar_power)
                epoch_val_pred += list(pred.cpu().detach().numpy())
                epoch_val_truth += list(solar_power.cpu().detach().numpy())

        val_len = len(validation_2011) + len(validation_2012)
        val_loss.append(epoch_val_loss/val_len)

        epoch_val_pred = torch.tensor(epoch_val_pred)
        epoch_val_truth = torch.tensor(epoch_val_truth)
        epoch_val_nMAP = torch.mean(torch.abs(epoch_val_pred-epoch_val_truth)/torch.mean(epoch_val_truth)*100)
        val_nMAP.append(epoch_val_nMAP.item())

    print(f"Finish epoch {epoch}.")
    print(f"Train_loss = {train_loss[-1]}. Train_nMAP = {train_nMAP[-1]}")
    print(f"Val_loss = {val_loss[-1]}. Val_nMAP = {val_nMAP[-1]}")

    # check_point = {
    #         "epoch": epoch,
    #         "alex_net": alex_net.state_dict().copy(),
    #         "alex_net_optim": optim.state_dict().copy(),
    #         "train_loss": train_loss.copy(),
    #         "val_loss": val_loss.copy(),
    # }

    # Save model
    save_path = "../checkpoint/alex_net_auxiliary/alex_net_auxiliary_{}epoch.pt".format(epoch)
    save_model(epoch=epoch, model=alex_net, optim=optim, train_loss=train_loss, val_loss=val_loss, save_path=save_path)
        
    # Save learning curve
    learning_curve_save_path = "../checkpoint/alex_net_auxiliary/learning_curve_{}epoch".format(epoch)
    save_learning_curve(train_loss, val_loss, learning_curve_save_path)

