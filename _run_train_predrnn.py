import os
import torch
import torchvision
import numpy as np
# from piq import ssim, psnr
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from utils import *
from dataset import SkyImagesDataset
from models import PredRNN

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
print(f"Device: {torch.cuda.device_count()} GPUs - {torch.cuda.get_device_name(0)}")
print()


# ***************************************************************************************************************
# ***************************************************************************************************************
# ************************************************* Init dataset ************************************************
# ***************************************************************************************************************
# ***************************************************************************************************************
print("Start loading dataset")
# Define forecast horizon
forecast_horizon = "4h"
train_batch_size = 8
val_batch_size = 64
# Init dataset
training_dir = "../processed_dataset/training/"
training_set = SkyImagesDataset(data_dir=training_dir, forecast_horizon=forecast_horizon)
training_set_dataloader = DataLoader(dataset=training_set, batch_size=train_batch_size, shuffle=True, num_workers=16)
# Validation 2011
validation_2011_dir = "../processed_dataset/validation_2011/"
validation_2011 = SkyImagesDataset(data_dir=validation_2011_dir, forecast_horizon=forecast_horizon)
validation_2011_dataloader = DataLoader(dataset=validation_2011, batch_size=val_batch_size, shuffle=False, num_workers=16)
# Validation 2012
validation_2012_dir = "../processed_dataset/validation_2012/"
validation_2012 = SkyImagesDataset(data_dir=validation_2012_dir, forecast_horizon=forecast_horizon)
validation_2012_dataloader = DataLoader(dataset=validation_2012, batch_size=val_batch_size, shuffle=False, num_workers=16)
# Test 2015
# test_2015_dir = "../processed_dataset/test_2015/"
# test_2015 = SkyImagesDataset(data_dir=test_2015_dir, forecast_horizon=forecast_horizon)
# test_2015_dataloader = DataLoader(dataset=test_2015, batch_size=batch_size, shuffle=False)
# # Test 2016
# test_2016_dir = "../processed_dataset/test_2016/"
# test_2016 = SkyImagesDataset(data_dir=test_2016_dir, forecast_horizon=forecast_horizon)
# test_2016_dataloader = DataLoader(dataset=test_2016, batch_size=batch_size, shuffle=False)


# ***************************************************************************************************************
# ***************************************************************************************************************
# ****************************************** Init model and parameters ******************************************
# ***************************************************************************************************************
# ***************************************************************************************************************
# Init model 
predrnn = PredRNN(nb_layers=3, image_shape=(128,128,3), in_channel=3, hidden_layer_dim=32, kernel_size=7, stride=1)
predrnn = torch.nn.DataParallel(predrnn, device_ids=[0,1,2,3])
predrnn.to(device=device)
optim = torch.optim.Adam(params=predrnn.parameters(), lr=0.003)


# Init loss
l1_loss = torch.nn.L1Loss()
# Init some variables for training
train_loss = []
val_loss = []
current_epoch = 0
nb_epochs = 5


# ******************************************* Load model (delte later) ******************************************
# load_epoch = 3
# checkpoint = torch.load(f"../checkpoint/predrnn_{forecast_horizon}/predrnn_{load_epoch}epoch.pt")
# predrnn.load_state_dict(checkpoint["model_state_dict"])
# optim.load_state_dict(checkpoint["optim_state_dict"])
# train_loss = checkpoint["train_loss"]
# val_loss = checkpoint["val_loss"]
# current_epoch = load_epoch + 1
# print(f"Load model at epoch {load_epoch} succesfully \n")
# ******************************************* Load model (delte later) ******************************************


# ***************************************************************************************************************
# ***************************************************************************************************************
# ************************************************* Train model *************************************************
# ***************************************************************************************************************
# ***************************************************************************************************************
for epoch in range(current_epoch, nb_epochs):
    print("Epoch", epoch)
    if epoch == 2:
        for g in optim.param_groups:
            g['lr'] = 0.001
    if epoch == 3:
        for g in optim.param_groups:
            g['lr'] = 0.0005
    # if epoch == 4:
    #     for g in optim.param_groups:
    #         g['lr'] = 0.0003

    # Training
    predrnn.train()
    epoch_train_loss = 0
    for i, sample in enumerate(tqdm(training_set_dataloader)):
        # if i < 12935:
        #     continue
        if i == 12939:
            print("break")
            break

        # Batch, length, channel, height, width
        _, input_image_sequence, target_image_sequence = sample
        input_image_sequence = input_image_sequence.float().to(device=device).permute(0,1,4,2,3) / 255.0
        target_image_sequence = target_image_sequence.float().to(device=device).permute(0,1,4,2,3) / 255.0
        target_image_sequence = torch.cat([input_image_sequence[:, 1:], target_image_sequence], axis=1)
        pred = predrnn(input_image_sequence, total_length=target_image_sequence.shape[1])
        pred = pred/2 + 0.5
        loss = l1_loss(pred, target_image_sequence)
        optim.zero_grad()
        loss.backward()
        optim.step()
        epoch_train_loss += loss.item() * input_image_sequence.shape[0]
        # ************************************ Fix later ************************************
        if i % 1000 == 0:
            grid = torchvision.utils.make_grid(input_image_sequence[0,:], nrow=4)
            grid = grid.permute(1,2,0).detach().cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.title("input")
            plt.imsave(f"../checkpoint/predrnn_{forecast_horizon}/input_{i}.png", grid)
            plt.clf()

            grid = torchvision.utils.make_grid(target_image_sequence[0,:], nrow=6)
            grid = grid.permute(1,2,0).detach().cpu().numpy()
            plt.figure(figsize=(15, 15))
            plt.title("ground_truth")
            plt.imsave(f"../checkpoint/predrnn_{forecast_horizon}/ground_truth_{i}.png", grid)
            plt.clf()

            grid = torchvision.utils.make_grid(pred[0,:], nrow=6)
            grid = grid.permute(1,2,0).detach().cpu().numpy()
            plt.figure(figsize=(15, 15))
            plt.title("prediction")
            plt.imsave(f"../checkpoint/predrnn_{forecast_horizon}/prediction_{i}.png", grid)
            plt.clf()
        # ************************************ Fix later ************************************


    train_loss.append(epoch_train_loss/len(training_set))
    # print("Finish epoch {}. Train_loss = {}".format(epoch, train_loss[-1]))
    save_path = f"../checkpoint/predrnn_{forecast_horizon}/predrnn_{epoch}epoch.pt"
    save_model(epoch=epoch, model=predrnn, optim=optim, train_loss=train_loss, val_loss=val_loss, save_path=save_path)


    # Validation
    predrnn.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for validation_set in [validation_2011_dataloader, validation_2012_dataloader]:
            for i, sample in enumerate(tqdm(validation_set)):
                # Batch, length, channel, height, width
                _, input_image_sequence, target_image_sequence = sample
                input_image_sequence = input_image_sequence.float().to(device=device).permute(0,1,4,2,3) / 255.0
                target_image_sequence = target_image_sequence.float().to(device=device).permute(0,1,4,2,3) / 255.0
                target_image_sequence = torch.cat([input_image_sequence[:, 1:], target_image_sequence], axis=1)
                pred = predrnn(input_image_sequence, total_length=target_image_sequence.shape[1])
                pred = pred/2 + 0.5
                loss = l1_loss(pred, target_image_sequence)
                epoch_val_loss += loss.item()*input_image_sequence.shape[0]

    val_loss.append(epoch_val_loss / (len(validation_2011) + len(validation_2012)))
    print(f"Finish epoch {epoch}. Train_loss = {train_loss[-1]}. Val_loss = {val_loss[-1]}")

    # Save model
    save_path = f"../checkpoint/predrnn_{forecast_horizon}/predrnn_{epoch}epoch.pt"
    save_model(epoch=epoch, model=predrnn, optim=optim, train_loss=train_loss, val_loss=val_loss, save_path=save_path)
        
    # Save learning curve
    learning_curve_save_path = f"../checkpoint/predrnn_{forecast_horizon}/learning_curve_{epoch}epoch"
    save_learning_curve(train_loss, val_loss, learning_curve_save_path)
    # Save validation image
    # image_save_path = f"../checkpoint/predrnn_{forecast_horizon}/image_{epoch}epoch"
    # save_image(input_image_sequence[0,:], target_image_sequence[0,:], pred[0,:], image_save_path)


