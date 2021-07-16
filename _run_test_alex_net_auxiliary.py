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

device = "cuda:2" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
print("Device: {} GPUs - {}".format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
print()


# ***************************************************************************************************************
# ***************************************************************************************************************
# ************************************************* Init dataset ************************************************
# ***************************************************************************************************************
# ***************************************************************************************************************
val_batch_size = 4096
# Init dataset
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
# # Init alex_net_auxiliary
auxiliary_indim = 7
alex_net_auxiliary = Alex_Net_Auxiliary(auxiliary_indim).to(device=device)
alex_net_auxiliary.to(device=device)
# Load alex_net_auxiliary
alex_net_auxiliary_load_epoch = 20
alex_net_auxiliary_checkpoint_path = f"../checkpoint/alex_net_auxiliary/alex_net_auxiliary_{alex_net_auxiliary_load_epoch}epoch.pt"
alex_net_auxiliary_checkpoint = torch.load(alex_net_auxiliary_checkpoint_path)
alex_net_auxiliary.load_state_dict(alex_net_auxiliary_checkpoint["model_state_dict"])
# alex_net_auxiliary = torch.nn.DataParallel(alex_net_auxiliary)
del alex_net_auxiliary_checkpoint

# Init loss
l1_loss = torch.nn.L1Loss()


alex_net_auxiliary.eval()
epoch_val_loss = 0
with torch.no_grad():
    for val_i, validation_loader in enumerate([test_2015_dataloader, test_2016_dataloader]):
        set_val_pred = []
        set_val_truth = []
        dataset_name = "test_2015" if val_i == 0 else "test_2016"
        for i, (image_id, image, auxiliary, solar_power) in enumerate(tqdm(validation_loader)):
            image = (image/255.0).to(device=device).permute(0,3,1,2)
            auxiliary = auxiliary.float().to(device=device)
            solar_power = solar_power.float().to(device=device)
            pred = alex_net_auxiliary(image, auxiliary).flatten()
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



