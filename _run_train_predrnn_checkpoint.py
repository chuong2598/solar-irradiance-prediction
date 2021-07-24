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

def save_model(epoch, trainstep, last_training_indices, data_loader, original_dataset_indices, resume_training_epoch,
               model, optim, epoch_train_loss, train_loss, val_loss, save_path):
    check_point = {
        "epoch": epoch, 
        "trainstep": trainstep,
        "last_training_indices": last_training_indices,
        "data_loader": data_loader,
        "original_dataset_indices": original_dataset_indices,
        "resume_training_epoch": resume_training_epoch,
        "epoch_train_loss": epoch_train_loss,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(check_point, save_path)
    model_name = save_path.split("/")[-1]
    print(f"Save model {model_name}") 

# !!!!!!!!!!!!!!!! Change here !!!!!!!!!!!!!!!!
def load_dataset(data_dir, forecast_horizon, manual_shuffle, batch_size):
    sequence_id = np.loadtxt(f"{data_dir}/sequence_id_{forecast_horizon}.txt", delimiter=",")
    indices = np.arange(len(sequence_id))
    if manual_shuffle == True:
        np.random.shuffle(indices)
    dataset = SkyImagesDataset(data_dir, indices, forecast_horizon=forecast_horizon)
    dataset_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    print(f"Load {data_dir} succesfully")
    return dataset, dataset_dataloader

class SkyImagesDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_dir, indices = [], forecast_horizon="10m"):
        """
        indices: Shufflted index list
        """
        assert forecast_horizon in ["10m", "1h", "4h"], ("Forecast horizon can be 10m, 1h, or 4h")
        self.sequence_id = np.loadtxt(f"{data_dir}/sequence_id_{forecast_horizon}.txt", delimiter=",")
        self.image_dir = f"{data_dir}/sky_images"

        self.last_training_indices = 0
        self.indices = indices

        # "trainstep": trainstep,
        # "last_training_indices": last_training_indices,
        # "data_loader": data_loader,
        # "original_dataset_indices": original_dataset_indices,
        # "resume_training_epoch": resume_training_epoch,
        # "epoch_train_loss": epoch_train_loss,

    def __len__(self):
        size = len(self.indices)
        return size

    def __getitem__(self, idx):
        self.last_training_indices = self.indices[idx]
        id_list = np.array(self.sequence_id[self.indices[idx]]).astype(int)

        images = []
        for image_id in id_list:
            image_id = str(int(image_id))
            # image = plt.imread(f"{self.image_dir}/{image_id[:8]}/{image_id}.raw.jpg")
            image = np.load(f"{self.image_dir}/{image_id[:8]}/{image_id}.raw.jpg.npy")
            # image = self.image_dict[image_id]
            images.append(image)
        images = np.array(images)
        input_image_sequence = images[:6]
        target_image_sequence = images[6:]
        return id_list, input_image_sequence, target_image_sequence


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
training_set, training_set_dataloader = load_dataset(training_dir, forecast_horizon, manual_shuffle=True, batch_size=train_batch_size)

# Validation 2011
validation_2011_dir = "../processed_dataset/validation_2011/"
validation_2011, validation_2011_dataloader = load_dataset(validation_2011_dir, forecast_horizon, manual_shuffle=False, batch_size=val_batch_size)

# Validation 2012
validation_2012_dir = "../processed_dataset/validation_2012/"
validation_2012, validation_2012_dataloader = load_dataset(validation_2012_dir, forecast_horizon, manual_shuffle=False, batch_size=val_batch_size)

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
trainstep = 0
resume_training_epoch = False


# ******************************************* Load model (delte later) ******************************************

# Load final checkpoint
load_epoch = 1
load_train_step = 61
epoch_complete = False
if epoch_complete:
    checkpoint = torch.load(f"../checkpoint/predrnn_{forecast_horizon}/predrnn_{load_epoch}epoch.pt")
    current_epoch = load_epoch + 1
else:
    checkpoint = torch.load(f"../checkpoint/predrnn_{forecast_horizon}/predrnn_{load_epoch}epoch_{load_train_step}_step.pt")
    current_epoch = load_epoch

predrnn.load_state_dict(checkpoint["model_state_dict"])
optim.load_state_dict(checkpoint["optim_state_dict"])
train_loss = checkpoint["train_loss"]
val_loss = checkpoint["val_loss"]

if load_epoch > 0:
    # Variable for continue training at the middle of the epoch
    trainstep = checkpoint["trainstep"]
    last_training_indices = checkpoint["last_training_indices"]
    training_set_dataloader = checkpoint["data_loader"]
    original_dataset_indices = checkpoint["original_dataset_indices"]
    resume_training_epoch = checkpoint["resume_training_epoch"]
    checkpoint_epoch_train_loss = checkpoint["epoch_train_loss"]

del checkpoint
print(f"Load model at epoch {load_epoch} succesfully \n")
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

    # !!!!!!!!!! if continue train => no shuffle !!!!!!!!!!
    if (resume_training_epoch):
        epoch_train_loss = checkpoint_epoch_train_loss
        print(f"Resume training at epoch = {epoch}, last_training_indices = {last_training_indices}")
        training_set_dataloader.dataset.indices = original_dataset_indices[list(original_dataset_indices).index(last_training_indices)+1:]
    else:
        epoch_train_loss = 0
        print(f"Start new training at epoch = {epoch}")
        training_set, training_set_dataloader = load_dataset(training_dir, forecast_horizon, manual_shuffle=True, batch_size=train_batch_size)
        original_dataset_indices = np.copy(training_set_dataloader.dataset.indices)

    # Training
    predrnn.train()
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
        pred = predrnn(input_image_sequence, total_length=target_image_sequence.shape[1], device=device)
        pred = pred/2 + 0.5
        loss = l1_loss(pred, target_image_sequence)
        optim.zero_grad()
        loss.backward()
        optim.step()
        epoch_train_loss += loss.item() * input_image_sequence.shape[0]
        # ************************************ Fix later ************************************
        if i % 10 == 0 and i > 0 :
            grid = torchvision.utils.make_grid(input_image_sequence[0,:], nrow=6)
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

            # Code to save checkpoint here
            trainstep = list(original_dataset_indices).index(training_set_dataloader.dataset.last_training_indices) // input_image_sequence.shape[0]
            save_path = f"../checkpoint/predrnn_{forecast_horizon}/predrnn_{epoch}epoch_{trainstep}_step.pt"
            save_model(epoch=epoch, trainstep=trainstep, last_training_indices=training_set_dataloader.dataset.last_training_indices, 
            data_loader=training_set_dataloader, original_dataset_indices=original_dataset_indices, resume_training_epoch=True,
               model=predrnn, optim=optim, epoch_train_loss=epoch_train_loss, train_loss=train_loss, val_loss=val_loss, save_path=save_path)

        # ************************************ Fix later ************************************


    train_loss.append(epoch_train_loss/len(training_set))
    # print("Finish epoch {}. Train_loss = {}".format(epoch, train_loss[-1]))
    save_path = f"../checkpoint/predrnn_{forecast_horizon}/predrnn_{epoch}epoch.pt"
    save_model(epoch=epoch, trainstep=-1, last_training_indices=-1, 
        data_loader=training_set_dataloader, original_dataset_indices=original_dataset_indices, resume_training_epoch=False,
        model=predrnn, optim=optim, epoch_train_loss=epoch_train_loss, train_loss=train_loss, val_loss=val_loss, save_path=save_path)
    resume_training_epoch = False

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
                pred = predrnn(input_image_sequence, total_length=target_image_sequence.shape[1], device=device)
                pred = pred/2 + 0.5
                loss = l1_loss(pred, target_image_sequence)
                epoch_val_loss += loss.item()*input_image_sequence.shape[0]

    val_loss.append(epoch_val_loss / (len(validation_2011) + len(validation_2012)))
    print(f"Finish epoch {epoch}. Train_loss = {train_loss[-1]}. Val_loss = {val_loss[-1]}")

    # Save model
    save_path = f"../checkpoint/predrnn_{forecast_horizon}/predrnn_{epoch}epoch.pt"
    save_model(epoch=epoch, trainstep=-1, last_training_indices=-1, 
        data_loader=training_set_dataloader, original_dataset_indices=original_dataset_indices, resume_training_epoch=False,
        model=predrnn, optim=optim, epoch_train_loss=epoch_train_loss, train_loss=train_loss, val_loss=val_loss, save_path=save_path)
    resume_training_epoch = False
        
    # Save learning curve
    learning_curve_save_path = f"../checkpoint/predrnn_{forecast_horizon}/learning_curve_{epoch}epoch"
    save_learning_curve(train_loss, val_loss, learning_curve_save_path)
    # Save validation image
    # image_save_path = f"../checkpoint/predrnn_{forecast_horizon}/image_{epoch}epoch"
    # save_image(input_image_sequence[0,:], target_image_sequence[0,:], pred[0,:], image_save_path)


