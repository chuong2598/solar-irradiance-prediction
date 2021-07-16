
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os
import collections
from tqdm.auto import tqdm
import pickle
import cv2
import torch
from utils import get_auxiliary_stats


class SkyImagesDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_dir, forecast_horizon="10m"):
        assert forecast_horizon in ["10m", "1h", "4h"], ("Forecast horizon can be 10m, 1h, or 4h")
        self.sequence_id = np.loadtxt(f"{data_dir}/sequence_id_{forecast_horizon}.txt", delimiter=",")
        self.image_dir = f"{data_dir}/sky_images"
        
        # self.image_dict = {}
        # folder_name_list = os.listdir(f"{data_dir}/sky_images")
        # for folder_name in tqdm(folder_name_list):
        #     if "DS" in folder_name: continue
        #     file_name_list = os.listdir(f"{data_dir}/sky_images/{folder_name}")
        #     for file_name in file_name_list:
        #         if file_name.endswith(".npy"):
        #             image = np.load("{}/{}/{}".format(self.image_dir, file_name[:8], file_name)).astype(int)
        #             self.image_dict[str(int(file_name[:14]))] = image
    
    def __len__(self):
        size = len(self.sequence_id)
        return size

    def __getitem__(self, idx):
        id_list = np.array(self.sequence_id[idx]).astype(int)
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
    

    
class SolarPowerDataset(torch.utils.data.Dataset):
    def __init__(self, samples_dir, dataset_name):
        self.samples_dir = samples_dir
        auxiliary_data = np.loadtxt(f"{samples_dir}/auxiliary_{dataset_name}.txt", delimiter=",")
        solar_power = auxiliary_data[:,6]
        self.image_indice = auxiliary_data[:,0]
        # Use all auxiliary_data (exclude index and solar power)
        auxiliary_data = auxiliary_data[:,[2,3,4,5,7,13,14]]
        # Get only data having valid solar power > 0
        valid_solar_power_indices = np.where(solar_power >= 0)[0]
        solar_power = solar_power[valid_solar_power_indices]
        auxiliary_data = auxiliary_data[valid_solar_power_indices]
        self.image_indice = self.image_indice[valid_solar_power_indices].astype(int)
        # Normalize auxiliary data
        # !!!!!! Fix mean later !!!!!!
        mean, std = get_auxiliary_stats()
        mean = mean[[0,6,7]]
        std = std[[0,6,7]]


        # mean, std = np.array([179.99055029, 59.09627829, 475.19600228])
        # std = np.array([65.78102436, 19.13922575, 287.84496435])
        auxiliary_data[:,-3:] = (auxiliary_data[:,-3:] - mean) / std
        
        self.auxiliary_dict = {}
        self.solar_power_dict = {}
        for i, image_id in enumerate(self.image_indice):
            self.auxiliary_dict[image_id] = auxiliary_data[i]
            self.solar_power_dict[image_id] = solar_power[i]
        
        image_indice_list = []
        folder_name_list = sorted(os.listdir(f"{samples_dir}/sky_images"))
        for folder_name in tqdm(folder_name_list):
            if "DS" in folder_name: continue
            file_name_list = sorted(os.listdir(f"{samples_dir}/sky_images/{folder_name}"))
            for file_name in file_name_list:
                if file_name.endswith(".npy"):
                    if (int(file_name[:14])) in self.image_indice:
                        image_indice_list.append(int(file_name[:14]))
        self.image_indice = np.array(image_indice_list)
        
            
    def __len__(self):
        size = len(self.image_indice)
        return size

    def __getitem__(self, idx):
        if(torch.is_tensor(idx)):
            idx = idx.tolist()
        image_id = int(self.image_indice[idx])
        # image = self.image_dict[image_id]
        file_name = str(image_id) + ".raw.jpg.npy"
        image = np.load(f"{self.samples_dir}/sky_images/{file_name[:8]}/{file_name}").astype(int)
        auxiliary =  self.auxiliary_dict[image_id]
        target_solar_power = self.solar_power_dict[image_id]
        return image_id, image, auxiliary, target_solar_power


class SkyImagesAuxiliaryDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset_name, forecast_horizon="10m"):
        assert forecast_horizon in ["10m", "1h", "4h"], ("Forecast horizon can be 10m, 1h, or 4h")
        # Get auxiliary dictionary
        auxiliary_data = np.loadtxt(f"{data_dir}/auxiliary_{dataset_name}.txt", delimiter=",")
        solar_power = auxiliary_data[:,6]
        # Use all auxiliary_data (exclude index and solar power)
        image_indice = auxiliary_data[:,0]
        auxiliary_data = auxiliary_data[:,[2,3,4,5,7,13,14]]
        # Get only data having valid solar power > 0
        valid_solar_power_indices = np.where(solar_power > 0)[0]
        image_indice = image_indice[valid_solar_power_indices]
        solar_power = solar_power[valid_solar_power_indices]
        auxiliary_data = auxiliary_data[valid_solar_power_indices]
        # Normalize auxiliary data
        mean, std = get_auxiliary_stats()
        mean = mean[[0,6,7]]
        std = std[[0,6,7]]
        # !!!!!! Fix mean later !!!!!!
        # mean = np.array([179.99055029, 59.09627829, 475.19600228])
        # std = np.array([65.78102436, 19.13922575, 287.84496435])
        auxiliary_data[:,-3:] = (auxiliary_data[:,-3:] - mean) / std
        # !!!!!! Fix mean later !!!!!!
        self.auxiliary_dict = {}
        self.solar_power_dict = {}
        for i, image_id in enumerate(image_indice):
            image_id = str(int(image_id))
            self.auxiliary_dict[image_id] = auxiliary_data[i]
            self.solar_power_dict[image_id] = solar_power[i]
        # Get image dictionary
        self.sequence_id = np.loadtxt(f"{data_dir}/sequence_id_auxiliary_{forecast_horizon}.txt", delimiter=",")
        self.image_dir = f"{data_dir}/sky_images"
        
#         self.image_dict = {}
#         folder_name_list = os.listdir(f"{data_dir}/sky_images")
        # for folder_name in tqdm(folder_name_list):
        #     if "DS" in folder_name: continue
        #     file_name_list = os.listdir(f"{data_dir}/sky_images/{folder_name}")
        #     for file_name in file_name_list:
        #         if file_name.endswith(".npy"):
        #             image = np.load("{}/{}/{}".format(self.image_dir, file_name[:8], file_name)).astype(int)
        #             # self.image_dict[str(int(file_name[:14]))] = image
        #             self.image_dict[str(int(file_name[:14]))] = -1
    
    def __len__(self):
        size = len(self.sequence_id)
        return size

    def __getitem__(self, idx):
        id_list = np.array(self.sequence_id[idx]).astype(int)
        images = []
        auxiliary_list = []
        solar_power_list = []
        for image_id in id_list:
            image_id = str(int(image_id))
            # image = self.image_dict[image_id]
            image = np.load("{}/{}/{}".format(self.image_dir, image_id[:8], image_id+".raw.jpg.npy")).astype(int)
            images.append(image)
            auxiliary_list.append(self.auxiliary_dict[image_id])
            solar_power_list.append(self.solar_power_dict[image_id])
        images = np.array(images)
        input_image_sequence = images[:6]
        target_image_sequence = images[6:]
        # Only get the first 4 auxiliary data as input
        auxiliary = np.array(auxiliary_list[:6])
        # Target solar power exclude the first two timestep
        solar_power = np.array(solar_power_list[1:])
        return id_list, input_image_sequence, target_image_sequence, auxiliary, solar_power


class AuxiliaryDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dataset_name, forecast_horizon="10m"):
        assert forecast_horizon in ["10m", "1h", "4h"], ("Forecast horizon can be 10m, 1h, or 4h")
        # Get auxiliary dictionary
        auxiliary_data = np.loadtxt(f"{data_dir}/auxiliary_{dataset_name}.txt", delimiter=",")
        solar_power = auxiliary_data[:,6]
        # Use all auxiliary_data (exclude index and solar power)
        image_indice = auxiliary_data[:,0]
        auxiliary_data = auxiliary_data[:,[2,3,4,5,7,13,14]]
        # Get only data having valid solar power > 0
        valid_solar_power_indices = np.where(solar_power > 0)[0]
        image_indice = image_indice[valid_solar_power_indices]
        auxiliary_data = auxiliary_data[valid_solar_power_indices]
        # Normalize auxiliary data
        mean, std = get_auxiliary_stats()
        mean = mean[[0,6,7]]
        std = std[[0,6,7]]
        # !!!!!! Fix mean later !!!!!!
        # mean = np.array([179.99055029, 59.09627829, 475.19600228])
        # std = np.array([65.78102436, 19.13922575, 287.84496435])
        # !!!!!! Fix mean later !!!!!!
        auxiliary_data[:,-3:] = (auxiliary_data[:,-3:] - mean) / std
        self.auxiliary_dict = {}
        for i, image_id in enumerate(image_indice):
            image_id = (int(image_id))
            self.auxiliary_dict[image_id] = auxiliary_data[i]
        # Get image dictionary
        self.sequence_id = np.loadtxt(f"{data_dir}/sequence_id_auxiliary_{forecast_horizon}.txt", delimiter=",")
        
    def __len__(self):
        size = len(self.sequence_id)
        return size

    def __getitem__(self, idx):
        id_list = np.array(self.sequence_id[idx]).astype(int)
        auxiliary_list = []
        for image_id in id_list:
            auxiliary_list.append(self.auxiliary_dict[image_id])
        # Only get the first 4 auxiliary data as input
        auxiliary = np.array(auxiliary_list[:6])
        target_auxiliary = np.array(auxiliary_list[6:])
        return id_list, auxiliary, target_auxiliary
