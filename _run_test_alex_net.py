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
from models import Alex_Net

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = "cuda:1" if torch.cuda.is_available() else "cpu"
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
# training_dir = "../processed_dataset/training/"
# training_set = SolarPowerDataset(samples_dir=training_dir, dataset_name="training")
# training_set_dataloader = DataLoader(dataset=training_set, batch_size=train_batch_size, shuffle=True)
# # Validation 2011
# validation_2011_dir = "../processed_dataset/validation_2011/"
# validation_2011 = SolarPowerDataset(samples_dir=validation_2011_dir, dataset_name="validation_2011")
# validation_2011_dataloader = DataLoader(dataset=validation_2011, batch_size=val_batch_size, shuffle=False)
# # Validation 2012
# validation_2012_dir = "../processed_dataset/validation_2012/"
# validation_2012 = SolarPowerDataset(samples_dir=validation_2012_dir, dataset_name="validation_2012")
# validation_2012_dataloader = DataLoader(dataset=validation_2012, batch_size=val_batch_size, shuffle=False)
# Test 2015
test_2015_dir = "../processed_dataset/test_2015/"
test_2015 = SolarPowerDataset(samples_dir=test_2015_dir, dataset_name="test_2015")
test_2015_dataloader = DataLoader(dataset=test_2015, batch_size=val_batch_size, shuffle=False)
# Test 2016
test_2016_dir = "../processed_dataset/test_2016/"
test_2016 = SolarPowerDataset(samples_dir=test_2016_dir, dataset_name="test_2016")
test_2016_dataloader = DataLoader(dataset=test_2016, batch_size=val_batch_size, shuffle=False)



# ***************************************************************************************************************
# ***************************************************************************************************************
# ****************************************** Init model and parameters ******************************************
# ***************************************************************************************************************
# ***************************************************************************************************************
# Init model 
alex_net = Alex_Net()
# alex_net = torch.nn.DataParallel(alex_net)
alex_net.to(device=device)
# Load alex_net
alex_net_load_epoch = 28
alex_net_checkpoint_path = f"../checkpoint/alex_net/alex_net_{alex_net_load_epoch}epoch.pt"
alex_net_checkpoint = torch.load(alex_net_checkpoint_path)
alex_net.load_state_dict(alex_net_checkpoint["model_state_dict"])

# Init loss
l1_loss = torch.nn.L1Loss()

# Evaluate on testing set
alex_net.eval()
epoch_val_loss = 0
with torch.no_grad():
    for val_i, validation_loader in enumerate([test_2015_dataloader, test_2016_dataloader]):
        set_val_pred = []
        set_val_truth = []
        dataset_name = "test_2015" if val_i == 0 else "test_2016"
        for i, (image_id, image, _, solar_power) in enumerate(tqdm(validation_loader)):
            image = (image/255.0).to(device=device).permute(0,3,1,2)
            solar_power = solar_power.float().to(device=device)
            pred = alex_net(image).flatten()
            loss = l1_loss(pred, solar_power)
            epoch_val_loss += loss.item()*len(solar_power)
            set_val_pred += list(pred.cpu().detach().numpy())
            set_val_truth += list(solar_power.cpu().detach().numpy())

        set_val_pred = torch.tensor(set_val_pred)
        set_val_truth = torch.tensor(set_val_truth)
        epoch_val_nMAP = torch.mean(torch.abs(set_val_pred-set_val_truth)/torch.mean(set_val_truth)*100)
        print(f"{dataset_name}:") 
        print(f"loss: {torch.mean(torch.abs(set_val_pred-set_val_truth))}")
        print(f"nMAP: {epoch_val_nMAP}")
        print() 

