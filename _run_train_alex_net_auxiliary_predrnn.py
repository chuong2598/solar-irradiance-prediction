import os
import torch
import torchvision
import numpy as np
# from piq import ssim, psnr
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from utils import *
from dataset import SkyImagesAuxiliaryDataset
from models import PredRNN, Auxiliary_LSTM, Alex_Net_Auxiliary

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
print("Device: {} GPUs - {}".format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
print()

# ***************************************************************************************************************
# ************************************************* Init dataset ************************************************
# ***************************************************************************************************************
# ***************************************************************************************************************
# Define forecast horizon
forecast_horizon = "1h"
train_batch_size = 16
val_batch_size = 64
# Init dataset
training_dir =  "../processed_dataset/training/"
training_set = SkyImagesAuxiliaryDataset(data_dir=training_dir, dataset_name="training", forecast_horizon=forecast_horizon)
training_set_dataloader = DataLoader(dataset=training_set, batch_size=train_batch_size, shuffle=True, num_workers = 16)
# Validation 2011
validation_2011_dir = "../processed_dataset/validation_2011/"
validation_2011 = SkyImagesAuxiliaryDataset(data_dir=validation_2011_dir, dataset_name="validation_2011", forecast_horizon=forecast_horizon)
validation_2011_dataloader = DataLoader(dataset=validation_2011, batch_size=val_batch_size, shuffle=False, num_workers = 16)
# Validation 2012
validation_2012_dir = "../processed_dataset/validation_2012/"
validation_2012 = SkyImagesAuxiliaryDataset(data_dir=validation_2012_dir, dataset_name="validation_2012", forecast_horizon=forecast_horizon)
validation_2012_dataloader = DataLoader(dataset=validation_2012, batch_size=val_batch_size, shuffle=False, num_workers = 16)

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
predrnn_load_epoch = 4
auxiliary_lstm_load_epoch = 49
alex_net_auxiliary_load_epoch = 20

# Init predrnn model 
predrnn = PredRNN(nb_layers=3, image_shape=(128,128,3), in_channel=3, hidden_layer_dim=32, kernel_size=7, stride=1)
predrnn = torch.nn.DataParallel(predrnn, device_ids=[0, 1, 2, 3])
predrnn.to(device=device)
# Load predrnn model
predrnn_checkpoint_path = f"../checkpoint/predrnn_{forecast_horizon}/predrnn_{predrnn_load_epoch}epoch.pt"
predrnn_checkpoint = torch.load(predrnn_checkpoint_path)
predrnn.load_state_dict(predrnn_checkpoint["model_state_dict"])
optim = torch.optim.Adam(params=predrnn.parameters(), lr=0.0005)
del predrnn_checkpoint

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
auxiliary_lstm = torch.nn.DataParallel(auxiliary_lstm, device_ids=[0, 1, 2, 3])
del auxiliary_lstm_checkpoint

# Init alex_net_auxiliary
alex_net_auxiliary = Alex_Net_Auxiliary(auxiliary_indim).to(device=device)
alex_net_auxiliary.to(device=device)
# Load alex_net_auxiliary
alex_net_auxiliary_checkpoint_path = f"../checkpoint/alex_net_auxiliary/alex_net_auxiliary_{alex_net_auxiliary_load_epoch}epoch.pt"
alex_net_auxiliary_checkpoint = torch.load(alex_net_auxiliary_checkpoint_path)
alex_net_auxiliary.load_state_dict(alex_net_auxiliary_checkpoint["model_state_dict"])
alex_net_auxiliary = torch.nn.DataParallel(alex_net_auxiliary, device_ids=[0, 1, 2, 3])
del alex_net_auxiliary_checkpoint

# !!! Freeze alex_net_auxiliary and auxiliary_lstm !!!
alex_net_auxiliary.eval()
auxiliary_lstm.eval()
# !!! Freeze alex_net_auxiliary and auxiliary_lstm !!!

# Init loss
l1_loss = torch.nn.L1Loss()
# Init some variables for training
train_loss = []
val_loss = []
current_epoch = 0
nb_epochs = 10


# ******************** Load checkpoint (delete later) ********************
# load_epoch = 0
# current_epoch = load_epoch + 1
# predrnn_checkpoint_path = f"../checkpoint/alex_net_auxiliary_predrnn_{forecast_horizon}/predrnn_{load_epoch}epoch_27139step.pt"
# predrnn_checkpoint = torch.load(predrnn_checkpoint_path)
# predrnn.load_state_dict(predrnn_checkpoint["model_state_dict"])
# optim.load_state_dict(predrnn_checkpoint["optim_state_dict"])
# for g in optim.param_groups:
#     g['lr'] = 0.0003
# del predrnn_checkpoint

# ***************************************************************************************************************
# ***************************************************************************************************************
# ************************************************* Train model *************************************************
# ***************************************************************************************************************
# ***************************************************************************************************************

for epoch in range(current_epoch, nb_epochs):
    print(f"Epoch {epoch}")

    # Training
    predrnn.train()
    epoch_train_loss = 0
    # Variable to check running loss
    # epoch_predrnn_loss = 0
    # epoch_alex_loss = 0
    for i, sample in enumerate(tqdm(training_set_dataloader)):
        # if i % 5000 == 0 or i == 27138 or i == 27139 or i == 27140:
        #     print(f"Saving model at step {i}....")
        #     train_loss.append(epoch_train_loss/len(training_set))
        #     save_path = f"../checkpoint/alex_net_auxiliary_predrnn_{forecast_horizon}/predrnn_{epoch}epoch_{i}step.pt"
        #     save_model(epoch=epoch, model=predrnn, optim=optim, train_loss=train_loss, val_loss=val_loss, save_path=save_path)
        #     print("Save success")
        # if i == 27139 or i == 27140:
        #     print(sample[0])
        if i == 9474: 
            break

        # Process data
        _, input_image_sequence, target_image_sequence, auxiliary, solar_power = sample
        input_image_sequence = input_image_sequence.float().to(device=device).permute(0,1,4,2,3) / 255.0
        target_image_sequence = target_image_sequence.float().to(device=device).permute(0,1,4,2,3) / 255.0
        target_image_sequence = torch.cat([input_image_sequence[:, 1:], target_image_sequence], axis=1)
        solar_power = (solar_power).float().to(device=device).view(-1,1)
        auxiliary = auxiliary.float().to(device=device)
        # PredRnn prediction
        image_pred = predrnn(input_image_sequence, total_length=target_image_sequence.shape[1])
        image_pred = image_pred/2 + 0.5
        predrnn_loss = l1_loss(image_pred, target_image_sequence)
        # Auxiliary prediction
        auxiliary_pred = auxiliary_lstm(auxiliary, total_length=target_image_sequence.shape[1])
        # Alex_net prediction
        image_pred = image_pred.contiguous().view(-1,3,128,128)
        auxiliary_pred = auxiliary_pred.contiguous().view(-1,auxiliary_indim)
        solar_pred = alex_net_auxiliary(image_pred, auxiliary_pred)
        alex_loss = l1_loss(solar_pred, solar_power)
        # Check running lost (in order to balance the two loss)
        # epoch_predrnn_loss += predrnn_loss.item()*len(input_image_sequence)
        # epoch_alex_loss += alex_loss.item()*len(input_image_sequence)
        # print(f"predrnn_loss: {epoch_predrnn_loss/ ((i+1)*len(input_image_sequence)) }")
        # print(f"alex_loss: {epoch_alex_loss/((i+1)*len(input_image_sequence))}")
        # print()

        # Calculate total loss and Update (change weight of alex_loss to 0.0005 (it was 0.0007 for 1 hour) )
        total_loss = predrnn_loss + alex_loss*0.0007
        optim.zero_grad()
        total_loss.backward()
        optim.step()
        # Accumulate loss
        epoch_train_loss += total_loss.item() * input_image_sequence.shape[0]
        # ************************************ Fix later ************************************
        if i % 1000 == 0:
            grid = torchvision.utils.make_grid(input_image_sequence[0,:], nrow=4)
            grid = grid.permute(1,2,0).detach().cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.title("input")
            plt.imsave(f"../checkpoint/alex_net_auxiliary_predrnn_{forecast_horizon}/input_{i}.png", grid)
            plt.clf()

            grid = torchvision.utils.make_grid(target_image_sequence[0,:], nrow=6)
            grid = grid.permute(1,2,0).detach().cpu().numpy()
            plt.figure(figsize=(15, 15))
            plt.title("ground_truth")
            plt.imsave(f"../checkpoint/alex_net_auxiliary_predrnn_{forecast_horizon}/ground_truth_{i}.png", grid)
            plt.clf()

            grid = torchvision.utils.make_grid(image_pred.view(target_image_sequence.shape)[0,:], nrow=6)
            grid = grid.permute(1,2,0).detach().cpu().numpy()
            plt.figure(figsize=(15, 15))
            plt.title("prediction")
            plt.imsave(f"../checkpoint/alex_net_auxiliary_predrnn_{forecast_horizon}/prediction_{i}.png", grid)
            plt.clf()
        # ************************************ Fix later ************************************


    train_loss.append(epoch_train_loss/len(training_set))
    # print("Finish epoch {}. Train_loss = {}".format(epoch, train_loss[-1]))
    save_path = f"../checkpoint/alex_net_auxiliary_predrnn_{forecast_horizon}/predrnn_{epoch}epoch.pt"
    save_model(epoch=epoch, model=predrnn, optim=optim, train_loss=train_loss, val_loss=val_loss, save_path=save_path)

    # Validation
    predrnn.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for validation_set in [validation_2011_dataloader, validation_2012_dataloader]:
            for i, sample in enumerate(tqdm(validation_set)):
                # Process data
                _, input_image_sequence, target_image_sequence, auxiliary, solar_power = sample
                input_image_sequence = input_image_sequence.float().to(device=device).permute(0,1,4,2,3) / 255.0
                target_image_sequence = target_image_sequence.float().to(device=device).permute(0,1,4,2,3) / 255.0
                target_image_sequence = torch.cat([input_image_sequence[:, 1:], target_image_sequence], axis=1)
                solar_power = (solar_power).float().to(device=device).view(-1,1)
                auxiliary = auxiliary.float().to(device=device)
                # PredRnn prediction
                image_pred = predrnn(input_image_sequence, total_length=target_image_sequence.shape[1])
                image_pred = image_pred/2 + 0.5
                predrnn_loss = l1_loss(image_pred, target_image_sequence)
                # Auxiliary prediction
                auxiliary_pred = auxiliary_lstm(auxiliary, total_length=target_image_sequence.shape[1])
                # Alex_net prediction
                image_pred = image_pred.contiguous().view(-1,3,128,128)
                auxiliary_pred = auxiliary_pred.contiguous().view(-1,auxiliary_indim)
                solar_pred = alex_net_auxiliary(image_pred, auxiliary_pred)
                alex_loss = l1_loss(solar_pred, solar_power)
                # Calculate total loss and Update
                total_loss = predrnn_loss + alex_loss*0.0007
                # Accumulate loss
                epoch_val_loss += total_loss.item() * input_image_sequence.shape[0]
            

    val_loss.append(epoch_val_loss/(len(validation_2011) + len(validation_2012)))
    print(f"Finish epoch {epoch}. Train_loss = {train_loss[-1]}. Val_loss = {val_loss[-1]}")

    # Save model
    save_path = f"../checkpoint/alex_net_auxiliary_predrnn_{forecast_horizon}/predrnn_{epoch}epoch.pt"
    save_model(epoch=epoch, model=predrnn, optim=optim, train_loss=train_loss, val_loss=val_loss, save_path=save_path)
    # Save learning curve
    learning_curve_save_path = f"../checkpoint/alex_net_auxiliary_predrnn_{forecast_horizon}/learning_curve_{epoch}epoch"
    save_learning_curve(train_loss, val_loss, learning_curve_save_path)
    # Save validation image
    # image_save_path = f"../checkpoint/alex_net_auxiliary_predrnn_{forecast_horizon}/image_{epoch}epoch"
    # save_image(input_image_sequence[0,:], target_image_sequence[0,:], pred[0,:], image_save_path)

